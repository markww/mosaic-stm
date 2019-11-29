use indexmap::IndexMap;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;
use std::ptr::NonNull;
use std::any::TypeId;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::spin_loop_hint;
use std::ops::Deref;
use std::ops::DerefMut;
use std::marker::PhantomData;
use std::alloc::Layout;
use std::alloc::System;
use std::alloc::GlobalAlloc;
use std::collections::HashMap;
use std::ptr::null_mut;
use lazy_static::lazy_static;
use std::thread_local;
use std::cell::RefCell;
use std::cell::RefMut;
use std::result::Result;
use std::u64::MAX;

// The inner form of a TVarVersion is the type itself plus a pointer to "next".
// These versions will be, at various stages in their lifetime, inserted into
// linked lists, and so we provide that extra pointer-width.
//
// We make this type have C representation so that we can access the
// write_txn_created field and the next_ptr field without knowledge of what the
// GuardedType is; that is, we can just fake it by use a pointer to a
// TVarVersionInner with a () GuardedType.
#[repr(C)]
struct TVarVersionInner<GuardedType> {
    write_txn_created: u64,
    next_ptr: Option<NonNull<TVarVersionInner<GuardedType>>>,
    inner_struct: GuardedType
}

#[derive(Copy, Clone)]
struct TypeIdAnnotatedPtr {
    type_id: TypeId,
    ptr: NonNull<()>
}

lazy_static! {
    static ref id_to_layout_map:
        RwLock<HashMap<TypeId, Layout>> = RwLock::new(HashMap::new());
}

fn idempotent_add_type_get_layout_and_id<T: 'static>() -> (TypeId, Layout) {
    let this_type_typeid = TypeId::of::<T>();
    let this_type_layout = Layout::new::<T>();
    // No matter what happens, if this function succeeds we will return these
    // items.
    let result_pair = (this_type_typeid, this_type_layout);
    {
        let read_lock_guard = id_to_layout_map.read().unwrap();
        let map_lookup_result = read_lock_guard.get(&this_type_typeid);
        match map_lookup_result {
            Some(layout) => {
                assert_eq!(*layout, this_type_layout);
                return result_pair;
            }
            None => {
                // Do nothing, we need to add this item using the write lock.
            }
        };
    }

    // If we reached this point, the entry was not present and we need to add
    // it.
    let write_lock_guard = &mut id_to_layout_map.write().unwrap();
    let layout_in_map =
        write_lock_guard
            .entry(this_type_typeid)
            .or_insert(this_type_layout);
    assert_eq!(*layout_in_map, this_type_layout);
    result_pair
}

fn get_layout_for_type_id_assert_present(typeid: TypeId) -> Layout {
    let read_lock_guard = id_to_layout_map.read().unwrap();
    read_lock_guard[&typeid]
}

// Why have this be u8? Because we want it to take up space in memory and thus
// have a unique address (I can't find a straight answer indicating whether
// pointers to zero sized types are guaranteed to have unique addresses or not,
// so better safe than sorry)
const zero_sized_type_representative: u8 = 0;

impl TypeIdAnnotatedPtr {

    fn alloc<T: 'static>(new_val: T) -> TypeIdAnnotatedPtr {
        let (type_id, layout) = idempotent_add_type_get_layout_and_id::<T>();
        let ptr =
            if layout.size() == 0 {
                NonNull::from(&zero_sized_type_representative).cast::<()>()
            } else {
                let new_ptr = unsafe { System.alloc(layout) };
                assert!(new_ptr != null_mut());
                unsafe { *(new_ptr as * mut T) = new_val };
                NonNull::new(new_ptr).unwrap().cast::<()>()
            };
        TypeIdAnnotatedPtr { type_id, ptr }
    }

    fn dealloc(&self) {
        unsafe {
            System.dealloc
                (self.ptr.cast::<u8>().as_ptr(),
                get_layout_for_type_id_assert_present(self.type_id));
        }
    }

    fn dynamic_cast<T: 'static>(self) -> Option<NonNull<T>> {
        let target_type_id = TypeId::of::<T>();
        if target_type_id == self.type_id {
            Some(self.ptr.cast::<T>())
        } else {
            None
        }
    }
}

#[derive(Copy, Clone)]
struct DynamicInnerTVarPtr(TypeIdAnnotatedPtr);

impl DynamicInnerTVarPtr {
    fn downcast_to_inner_version_having_guarded<GuardedType: 'static>(&self) ->
        NonNull<TVarVersionInner<GuardedType>>
    {
        self.0.dynamic_cast::<TVarVersionInner<GuardedType>>().unwrap()
    }

    fn get_guarded_ref<GuardedType: 'static>(&self) -> &'static GuardedType {
        &(
            unsafe {
                &*self
                    .downcast_to_inner_version_having_guarded::<GuardedType>().as_ptr()
            }
        ).inner_struct
    }

    fn get_guarded_mut<GuardedType: 'static>(&mut self)
        -> &'static mut GuardedType
    {
        &mut (
            unsafe {
                &mut *self
                    .downcast_to_inner_version_having_guarded::<GuardedType>().as_ptr()
            }
        ).inner_struct
    }

    fn wrap_static_inner_version<GuardedType: 'static>
        (nonnull_inner_version: NonNull<TVarVersionInner<GuardedType>>)
        -> DynamicInnerTVarPtr
    {
        DynamicInnerTVarPtr {
            0: TypeIdAnnotatedPtr {
                ptr: nonnull_inner_version.cast::<()>(),
                type_id: TypeId::of::<TVarVersionInner<GuardedType>>()
            }
        }
    }

    fn alloc<GuardedType: 'static>(new_val: GuardedType)
        -> DynamicInnerTVarPtr
    {
        DynamicInnerTVarPtr {
            0: TypeIdAnnotatedPtr::alloc(TVarVersionInner {
                write_txn_created: 0,
                next_ptr: None,
                inner_struct: new_val
            })
        }
    }

    fn dealloc(&mut self) {
        self.0.dealloc()
    }

    fn has_guarded_type<GuardedType: 'static>(&self) -> bool {
        TypeId::of::<TVarVersionInner<GuardedType>>() == self.0.type_id
    }
}

#[derive(Copy, Clone)]
struct TVarImmVersion<GuardedType> {
    type_erased_inner: DynamicInnerTVarPtr,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType: 'static> TVarImmVersion<GuardedType> {
    fn wrap_inner_version_ptr
        (inner_version_ptr: NonNull<TVarVersionInner<GuardedType>>)
        -> TVarImmVersion<GuardedType>
    {
        TVarImmVersion {
            type_erased_inner:
                DynamicInnerTVarPtr::wrap_static_inner_version
                    (inner_version_ptr),
            phantom: PhantomData
        }
    }

    fn get_dyn_inner_version_ptr(&self) -> DynamicInnerTVarPtr {
        self.type_erased_inner
    }
}

// A TVarShadowVersion represents a version of the object that is local to a
// transaction and which is still mutable. If the transaction fails and
// restarts, then this version will be dropped. If the transaction succeeds,
// then this will be converted to a TVarImmVersion so that it can be shared
// between threads without worrying about mutation.
#[derive(Clone, Copy)]
struct TVarShadowVersion<GuardedType> {
    type_erased_inner: DynamicInnerTVarPtr,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType: Copy + 'static> TVarShadowVersion<GuardedType> {

    fn new(guarded: GuardedType) -> TVarShadowVersion<GuardedType>
    {
        TVarShadowVersion {
            type_erased_inner: DynamicInnerTVarPtr::alloc(guarded),
            phantom: PhantomData
        }
    }

    fn copy_from_imm_version
        (imm_version: TVarImmVersion<GuardedType>) ->
            TVarShadowVersion<GuardedType>
    {
        Self::new(*imm_version.get_dyn_inner_version_ptr().get_guarded_ref())
    }

    fn get_dyn_inner_version_ptr(&self) -> DynamicInnerTVarPtr {
        self.type_erased_inner
    }
}

// This special value indicates that the pointer is temporarily reserved for a
// write commit. This causes new transactions that wish to grab the canonical
// value to spin.
const reserved_for_commit_representative: u8 = 0;
const reserved_for_commit_ptr: * const () =
    (&reserved_for_commit_representative as * const u8) as * const ();

enum TVarCanonPtrContents {
    ReservedForCommit,
    Available(NonNull<()>)
}

impl TVarCanonPtrContents {
    fn convert_raw_ptr_to_current_version_or_status(raw_ptr: * mut ())
        -> TVarCanonPtrContents
    {
        if (raw_ptr as * const ()) == reserved_for_commit_ptr {
            TVarCanonPtrContents::ReservedForCommit
        } else {
            TVarCanonPtrContents::Available
                (NonNull::new(raw_ptr).unwrap())
        }
    }
}

struct VersionedTVarTypeErased {
    canon_ptr: AtomicPtr<()>,
    stale_version_list: AtomicPtr<()>,
    type_id: TypeId
}

pub struct VersionedTVar<GuardedType> {
    inner_type_erased: VersionedTVarTypeErased,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType : Copy + 'static> VersionedTVar<GuardedType> {
    fn new(inner_val: GuardedType) -> VersionedTVar<GuardedType> {
        // To use the same flow of functions that a regular update to the TVar
        // uses, just create a shadow version and turn it into a TVarImmVersion.

        let init_shadow_version = TVarShadowVersion::new(inner_val);

        let init_dynamic_inner_version_ptr =
            init_shadow_version.type_erased_inner;
        let result: VersionedTVar<GuardedType> =
            VersionedTVar {
                inner_type_erased: VersionedTVarTypeErased {
                    canon_ptr:
                        AtomicPtr::new
                            (init_dynamic_inner_version_ptr.0.ptr.as_ptr()),
                    stale_version_list: AtomicPtr::new(null_mut()),
                    type_id: init_dynamic_inner_version_ptr.0.type_id
                },
                phantom: PhantomData
            };
        result
    }

    // This function gets the current canon version or, if a special status code
    // pointer is in the canon pointer, the status.
    fn get_current_canon_version_or_status(&self)
        -> TVarCanonPtrContents
    {
        let raw_canon_ptr =
            self.inner_type_erased.canon_ptr.load(Acquire);
        TVarCanonPtrContents::convert_raw_ptr_to_current_version_or_status
            (raw_canon_ptr)
    }

    // This function spins until it gets the current canon version.
    fn get_current_canon_version(&self) -> TVarImmVersion<GuardedType> {
        // null is used to indicate that the canonical version is undergoing an
        // update but is mid-transaction. If we see this, spin until we get a
        // non-null value.
        let type_id = self.inner_type_erased.type_id;
        loop {
            match self.get_current_canon_version_or_status() {
                TVarCanonPtrContents::ReservedForCommit => {
                    spin_loop_hint();
                    continue;
                },
                TVarCanonPtrContents::Available(ptr) => {
                    let type_id_annot_ptr = TypeIdAnnotatedPtr { ptr, type_id };
                    let dynamic_inner_version_ptr = DynamicInnerTVarPtr {
                        0: type_id_annot_ptr
                    };
                    return TVarImmVersion {
                        type_erased_inner: dynamic_inner_version_ptr,
                        phantom: PhantomData
                    }
                }
            }
        }
    }
}

struct CapturedTVarTypeErased {
    captured_version_ptr: NonNull<()>,
    shadow_copy_ptr: Option<NonNull<()>>,
    type_id: TypeId,
    // The nested transaction num which captured this tvar originally.
    nested_txn_num: u64
}

impl Drop for CapturedTVarTypeErased {
    fn drop(&mut self) {
        match self.shadow_copy_ptr {
            Some(present_shadow_copy) => {
                TypeIdAnnotatedPtr {
                    ptr: present_shadow_copy,
                    type_id: self.type_id
                }.dealloc();
            },
            None => {
                // Nothing to drop
            }
        }
    }
}

impl CapturedTVarTypeErased {

    fn new<GuardedType: 'static>
        (captured_version: TVarImmVersion<GuardedType>,
         txn_num: u64)
        -> CapturedTVarTypeErased
    {
        let dyn_inner_version_ptr =
            captured_version.get_dyn_inner_version_ptr();
        CapturedTVarTypeErased {
            captured_version_ptr: dyn_inner_version_ptr.0.ptr,
            type_id: dyn_inner_version_ptr.0.type_id,
            shadow_copy_ptr: None,
            nested_txn_num: txn_num
        }
    }

    fn get_captured_version<GuardedType: 'static>(&self)
         -> TVarImmVersion<GuardedType>
    {
        let result_dynamic_inner_version_ptr =
            DynamicInnerTVarPtr {
                0: TypeIdAnnotatedPtr {
                    type_id: self.type_id,
                    ptr: self.captured_version_ptr
                }
            };
        assert!(result_dynamic_inner_version_ptr
            .has_guarded_type::<GuardedType>());
        let result = TVarImmVersion {
            type_erased_inner: result_dynamic_inner_version_ptr,
            phantom: PhantomData
        };
        result
    }

    fn get_optional_shadow_copy<GuardedType: 'static>(&self)
        -> Option<TVarShadowVersion<GuardedType>>
    {
        match self.shadow_copy_ptr {
            Some(present_shadow_copy_ptr) => {
                let result_dynamic_inner_version_ptr = DynamicInnerTVarPtr {
                    0: TypeIdAnnotatedPtr {
                        ptr: present_shadow_copy_ptr,
                        type_id: self.type_id
                    }
                };
                assert!(result_dynamic_inner_version_ptr
                            .has_guarded_type::<GuardedType>());
                Some(TVarShadowVersion {
                    type_erased_inner: result_dynamic_inner_version_ptr,
                    phantom: PhantomData
                })
            },
            None => None
        }
    }

    fn get_shadow_copy_create_if_not_present<GuardedType: Copy + 'static>
        (&mut self) -> TVarShadowVersion<GuardedType>
    {
        match self.get_optional_shadow_copy() {
            Some(already_present_shadow_copy) => already_present_shadow_copy,
            None => {
                let captured_version =
                    self.get_captured_version::<GuardedType>();
                let new_shadow_version =
                    TVarShadowVersion::new
                        (*captured_version
                            .get_dyn_inner_version_ptr()
                            .get_guarded_ref());
                self.shadow_copy_ptr =
                    Some(new_shadow_version.type_erased_inner.0.ptr);
                new_shadow_version
            }
        }
    }

    fn get_guarded_ref<GuardedType: 'static + Copy>(&self) -> &GuardedType {
        let opt_shadow_copy_ref =
            self.get_optional_shadow_copy::<GuardedType>();
        match opt_shadow_copy_ref {
            Some(present_shadow_copy) =>
                present_shadow_copy
                    .get_dyn_inner_version_ptr()
                    .get_guarded_ref(),
            None =>
             self.get_captured_version::<GuardedType>()
                .get_dyn_inner_version_ptr()
                .get_guarded_ref()
        }
    }

    fn get_guarded_mut<GuardedType: 'static + Copy>(&mut self)
        -> &mut GuardedType
    {
        self.get_shadow_copy_create_if_not_present::<GuardedType>()
            .get_dyn_inner_version_ptr()
            .get_guarded_mut()
    }
}

pub struct CapturedTVarRef<'a, GuardedType> {
    captured_tvar_ref: &'a CapturedTVarTypeErased,
    phantom: PhantomData<GuardedType>
}

impl<'a, GuardedType: Copy + 'static> Deref for
    CapturedTVarRef<'a, GuardedType>
{
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        self.captured_tvar_ref.get_guarded_ref()
    }
}

pub struct CapturedTVarMut<'a, GuardedType> {
    captured_tvar_mut: &'a mut CapturedTVarTypeErased,
    phantom: PhantomData<GuardedType>
}

impl<'a, GuardedType: Copy + 'static> Deref for
    CapturedTVarMut<'a, GuardedType>
{
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        self.captured_tvar_mut.get_guarded_ref()
    }
}

impl<'a, GuardedType: Copy + 'static> DerefMut for
    CapturedTVarMut<'a, GuardedType>
{
    fn deref_mut(&mut self) -> &mut GuardedType {
        self.captured_tvar_mut.get_guarded_mut()
    }
}

lazy_static! {
    static ref next_write_txn_num: AtomicU64 = AtomicU64::new(1);
}

struct PerThreadTransactionState {
    // This associates tvars with captures so that we can look up the captured
    // values without constantly grabbing locks or using atomic operations.
    // We place this in the per-thread state because IndexMap will dynamically
    // allocate memory as it grows, but dynamic allocation performs
    // synchronization. By storing one map for reuse by different transactions,
    // we can avoid performing synchronization due to dynamic allocation in what
    // should be a lock-free algorithm.
    // We use IndexMap rather than BTreeMap because you can clear an IndexMap
    // without losing the reserved space. This does not appear to be the case
    // for BTreeMap. In addition, you can sort an IndexMap, which we will need
    // to do to update the tvars in a canonical order to avoid deadlocking.
    captured_tvar_cache:
        IndexMap<NonNull<VersionedTVarTypeErased>, CapturedTVarTypeErased>
}

impl PerThreadTransactionState {
    fn clear_txn_state(&mut self) {
        self.captured_tvar_cache.clear();
    }
}

thread_local! {
    static per_thread_txn_state: RefCell<PerThreadTransactionState> =
        RefCell::new(
            PerThreadTransactionState {
                captured_tvar_cache: IndexMap::new()
            });
}

pub struct TxnErrStatus {
    failed_nested_txn_num: u64
}

pub struct VersionedTransaction<'a> {
    per_thread_txn_state: &'a mut PerThreadTransactionState,
    nested_txn_num: u64,
    is_write_txn: bool,
    need_rollback: bool
}

impl<'a> Drop for VersionedTransaction<'a> {
    fn drop(&mut self) {
        self.per_thread_txn_state.clear_txn_state();
    }
}

static write_txn_count: AtomicU64 = AtomicU64::new(0);
const TXN_COUNTER_UPDATING_VAL: u64 = MAX;

impl<'a> VersionedTransaction<'a> {

    fn perform_txn<UserResult>
        (per_thread_txn_ref: RefMut<PerThreadTransactionState>,
         txn_fn: fn(&mut VersionedTransaction)
            -> Result<UserResult, TxnErrStatus>) -> UserResult
    {
        let mut per_thread_txn_ref_mut = per_thread_txn_ref;
        'txn_retry_loop: loop {
            let mut this_txn =
                VersionedTransaction {
                    per_thread_txn_state: per_thread_txn_ref_mut.deref_mut(),
                    nested_txn_num: 0,
                    is_write_txn: false,
                    need_rollback: false
                };
            let txn_fn_result = txn_fn(&mut this_txn);
            if txn_fn_result.is_err() {
                continue 'txn_retry_loop;
            }

            let user_result = txn_fn_result.ok().unwrap();

            // Why have this? It's technically possible for the user to
            // ignore all errors that we throw at them and continue to
            // completing the transaction "successfully". This catches them
            // even if the transaction was a read-only transaction.
            if this_txn.need_rollback {
                continue 'txn_retry_loop;
            }

            // We have successfully completed the user's function without
            // hitting any errors. Now we have to perform the commit action.
            // If this transaction was a read transaction, nothing more to do.
            if !this_txn.is_write_txn {
                return user_result;
            }

            let captured_tvar_cache =
                &mut this_txn.per_thread_txn_state.captured_tvar_cache;

            // If this is a write transaction, on the other hand, we have a
            // much more complicated commit process to perform. First, we
            // must swap the special write-reserve value in for all tvars
            // considered. If we encounter any unexpected values along the
            // way, another write must have slipped by us and we must abort.
            captured_tvar_cache.sort_keys();
            let mut reserve_for_write_iter =
                captured_tvar_cache.iter().enumerate();

            // This is the length of the region of the map that was
            // successfully swapped. Note that this is a length, not an idx,
            // so that if no swaps were successful, we can indicate that
            // with a swap length of 0.
            let mut successful_swap_length: usize = 0;
            let mut saw_conflict = false;
            'reserve_commit:
                while let Some((idx, (tvar_ptr, captured_tvar))) =
                    reserve_for_write_iter.next()
            {
                let expected_canon_ptr =
                    captured_tvar.captured_version_ptr.as_ptr();
                let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
                'cmp_ex_loop: loop {
                    // We are swapping in a marker pointer with no contents,
                    // and whether we fail or succeed, we care only about
                    // the pointer values, not any associated data. I think
                    // we can get away with Relaxed ordering here.
                    let cmp_ex_result =
                        tvar_ref.canon_ptr.compare_exchange_weak
                            (expected_canon_ptr,
                            reserved_for_commit_ptr as * mut (),
                            Relaxed,
                            Relaxed);
                    match cmp_ex_result {
                        Ok(_) => {
                            // Remember, this is successful swap *length*, so
                            // we want 1 plus the idx.
                            successful_swap_length = idx + 1;
                            continue 'reserve_commit;
                        },
                        Err(found_ptr) => {
                            let curr_version_or_status =
                                TVarCanonPtrContents::
                                convert_raw_ptr_to_current_version_or_status
                                (found_ptr);
                            match curr_version_or_status {
                                TVarCanonPtrContents::ReservedForCommit => {
                                    spin_loop_hint();
                                    continue 'cmp_ex_loop;
                                },
                                TVarCanonPtrContents::Available
                                    (available_ptr) =>
                                {
                                    let raw_available_ptr =
                                        available_ptr.as_ptr();
                                    if raw_available_ptr == expected_canon_ptr {
                                        // If this is the case, compare
                                        // exchange weak experienced a
                                        // spurious failure. Try again.
                                        continue 'cmp_ex_loop;
                                    } else {
                                        // We saw another pointer, and have
                                        // encountered a conflict.
                                        saw_conflict = true;
                                        break 'reserve_commit;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // If we saw a swap failure, we have to put everything back as
            // we found it and roll back the transaction.
            if saw_conflict {
                for (idx, (tvar_ptr, captured_tvar)) in
                    captured_tvar_cache.iter().enumerate()
                {
                    if idx == successful_swap_length {
                        continue 'txn_retry_loop;
                    }
                    let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
                    let original_canon_ptr =
                        captured_tvar.captured_version_ptr.as_ptr();

                    // We can use relaxed here because the original store
                    // that made this canonical in the first place used
                    // Release, so any Acquire load on this location will
                    // get the updated contents from happening-after that
                    // store.
                    tvar_ref.canon_ptr.store(original_canon_ptr, Relaxed);
                }
                // The above loop should always take the continue branch
                // eventually, so this should be unreachable.
                unreachable!();
            }

            // We have reached the point where the write transaction is
            // guaranteed to succeed.

            // If we did not see a swap failure, then we have successfully
            // reserved all of the tvars. Now, we must reserve our slot in the
            // canonical write ordering. Swap the reserved COUNTER_UPDATING
            // value into the txn write counter.
            //
            // XXX: This is a dumb approach, this causes all writing
            // transactions to serialize on the counter. We should change this
            // to instead have each inner version have a pointer to its counter
            // val, have that set to the counter updating initially, swap them
            // all in, perform a RELEASE write on the txn counter, and write
            // that into the pointed-to version. That will ensure that anyone
            // getting a particular counter val with ACQUIRE will see all writes
            // at and before that counter val.
            let mut this_write_counter_val = TXN_COUNTER_UPDATING_VAL;
            loop {
                let prev_counter_val =
                    write_txn_count.swap(TXN_COUNTER_UPDATING_VAL, Relaxed);
                // If another thread has already reserved the counter, we gotta
                // spin til they release it.
                if prev_counter_val == TXN_COUNTER_UPDATING_VAL {
                    spin_loop_hint();
                    continue;
                }
                let new_counter_val = prev_counter_val + 1;
                // It's highly unlikely we'll have 2^64 - 1 successful write
                // txns.
                assert!(new_counter_val == TXN_COUNTER_UPDATING_VAL);
                this_write_counter_val = new_counter_val;
                break;
            }

            // Now we must update all of the tvars we have touched. For those
            // that we did not modify, we must swap in the old version. For
            // those that we did, we must swap in our shadow version upgraded to
            // a new immutable version. Might as well use drain and empty the
            // map at the same time.
            //
            // XXX: Have to add the replaced canon pointers to the stale version
            // list.
            for (tvar_ptr, captured_tvar) in captured_tvar_cache.drain(..) {
                let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
                match captured_tvar.shadow_copy_ptr {
                    Some(shadow_copy_ptr) => {
                        // The shadow copy is becoming canonical, so don't run
                        // the captured tvar drop function (which would delete
                        // it).
                        std::mem::forget(captured_tvar);

                        let raw_shadow_copy_ptr = shadow_copy_ptr.as_ptr();
                        let ptr_to_shadow_copy_header =
                            raw_shadow_copy_ptr as * mut TVarVersionInner<()>;
                        unsafe {
                            (*ptr_to_shadow_copy_header).write_txn_created =
                                this_write_counter_val;
                        }
                        // This is our first time publishing this shadow version
                        // to other threads, so we must use a release ordering.
                        tvar_ref.canon_ptr.store(raw_shadow_copy_ptr, Release);
                    },
                    None => {
                        // If we are just replacing a pointer that was already
                        // canonical, we can use a Relaxed ordering to swap it
                        // in, as the pointed-to contents were already published
                        // to this point by an earlier release.
                        tvar_ref.canon_ptr.store
                            (captured_tvar.captured_version_ptr.as_ptr(),
                             Relaxed);
                    }
                }
            }

            // XXX: Have we done everything we need to do with the counter at
            // this point?

            return user_result;
        }
    }

    pub fn with<UserResult>
        (txn_fn: fn(&mut VersionedTransaction)
            -> Result<UserResult, TxnErrStatus>)
        -> UserResult
    {
        per_thread_txn_state.with(|per_thread_txn_state_key| {
            VersionedTransaction::perform_txn
                (per_thread_txn_state_key.borrow_mut(), txn_fn)
        })
    }

    fn hit_cache_with_tvar<GuardedType: Copy + 'static>
        (&mut self,
         tvar: &VersionedTVar<GuardedType>) -> &mut CapturedTVarTypeErased
    {
        let nested_txn_num = self.nested_txn_num;
        self.per_thread_txn_state
            .captured_tvar_cache
            .entry(NonNull::from(&tvar.inner_type_erased))
            .or_insert_with
                (||
                    CapturedTVarTypeErased::new
                    (tvar.get_current_canon_version(),
                        nested_txn_num))
    }

    pub fn get_ref_for_tvar<GuardedType: Copy + 'static>
        (&mut self,
         tvar: &VersionedTVar<GuardedType>)
        -> CapturedTVarRef<GuardedType>
    {
        CapturedTVarRef {
            captured_tvar_ref: self.hit_cache_with_tvar(tvar),
            phantom: PhantomData
        }
    }

    pub fn get_mut_for_tvar<GuardedType: Copy + 'static>
        (&mut self,
         tvar: &VersionedTVar<GuardedType>)
        -> CapturedTVarMut<GuardedType>
    {
        self.is_write_txn = true;
        CapturedTVarMut {
            captured_tvar_mut: self.hit_cache_with_tvar(tvar),
            phantom: PhantomData
        }
    }
}
