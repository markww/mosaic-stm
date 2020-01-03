use indexmap::IndexMap;
use indexmap::map::Entry::Occupied;
use indexmap::map::Entry::Vacant;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::RwLock;
use std::ptr::NonNull;
use std::any::TypeId;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::spin_loop_hint;
use std::marker::PhantomData;
use std::alloc::Layout;
use std::alloc::GlobalAlloc;
use std::collections::HashMap;
use std::ptr::null_mut;
use std::ptr::null;
use lazy_static::lazy_static;
use std::thread_local;
use std::cell::RefCell;
use std::cell::Cell;
use std::result::Result;
use std::u64::MAX;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ops::DerefMut;

// We use Jemalloc as an allocator because it does not synchronize on every
// allocation/deallocation, unlike regular C malloc/free. Jemalloc keeps a
// thread-local cache of objects, which is very nice for use with our algorithm
// that attempts to reduce numbers of synchronizations.
use jemallocator::Jemalloc;

static TXN_WRITE_TIME: AtomicU64 = AtomicU64::new(0);

// A magic number indicating that the transaction counter is in the middle of
// updating.
const TXN_COUNTER_UPDATING_VAL: u64 = MAX;
// A magic number indicating that this version does not exist at a canonical
// time. This is the timestamp value on shadow versions.
const TXN_COUNTER_NON_CANON: u64 = TXN_COUNTER_UPDATING_VAL - 1;

// A timestamp for values resulting from initialization. They are given the
// reserved value of 0, and thus are smaller than any other timestamp.
const TXN_COUNTER_INIT: u64 = 0;

#[derive(Clone)]
struct TxnTimestamp(Arc<AtomicU64>);

impl TxnTimestamp {
        fn new_updating() -> TxnTimestamp {
            TxnTimestamp {
                0: Arc::new(AtomicU64::new(TXN_COUNTER_UPDATING_VAL))
        }
    }
}

lazy_static! {
    static ref NON_CANON_TIMESTAMP: TxnTimestamp = TxnTimestamp {
        0: Arc::new(AtomicU64::new(TXN_COUNTER_NON_CANON))
    };

    static ref INIT_TIMESTAMP: TxnTimestamp = TxnTimestamp {
        0: Arc::new(AtomicU64::new(TXN_COUNTER_INIT))
    };
}

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
    timestamp: TxnTimestamp,
    next_ptr: * const (),
    inner_struct: GuardedType
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct TypeIdAnnotatedPtr {
    type_id: TypeId,
    ptr: NonNull<()>
}

lazy_static! {
    static ref ID_TO_LAYOUT_MAP:
        RwLock<HashMap<TypeId, Layout>> = RwLock::new(HashMap::new());
}

fn idempotent_add_type_get_layout_and_id<T: 'static>() -> (TypeId, Layout) {
    let this_type_typeid = TypeId::of::<T>();
    let this_type_layout = Layout::new::<T>();
    // No matter what happens, if this function succeeds we will return these
    // items.
    let result_pair = (this_type_typeid, this_type_layout);
    {
        let read_lock_guard = ID_TO_LAYOUT_MAP.read().unwrap();
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
    let write_lock_guard = &mut ID_TO_LAYOUT_MAP.write().unwrap();
    let layout_in_map =
        write_lock_guard
            .entry(this_type_typeid)
            .or_insert(this_type_layout);
    assert_eq!(*layout_in_map, this_type_layout);
    result_pair
}

fn get_layout_for_type_id_assert_present(typeid: TypeId) -> Layout {
    let read_lock_guard = ID_TO_LAYOUT_MAP.read().unwrap();
    read_lock_guard[&typeid]
}

// Why have this be u8? Because we want it to take up space in memory and thus
// have a unique address (I can't find a straight answer indicating whether
// pointers to zero sized types are guaranteed to have unique addresses or not,
// so better safe than sorry)
const ZERO_SIZED_TYPE_REPRESENTATIVE: u8 = 0;

impl TypeIdAnnotatedPtr {

    fn alloc<T: 'static>(new_val: T) -> TypeIdAnnotatedPtr {
        let (type_id, layout) = idempotent_add_type_get_layout_and_id::<T>();
        let ptr =
            if layout.size() == 0 {
                NonNull::from(&ZERO_SIZED_TYPE_REPRESENTATIVE).cast::<()>()
            } else {
                // The layout should have enough space to store a T, otherwise
                // we are in for an overflow.
                assert!(layout.size() >= std::mem::size_of::<T>());
                let new_ptr =
                    unsafe { Jemalloc.alloc(layout) } as * mut MaybeUninit<T>;
                assert!(new_ptr != null_mut());
                unsafe {
                    *new_ptr = MaybeUninit::new(new_val);
                }
                NonNull::new(new_ptr).unwrap().cast::<()>()
            };
        TypeIdAnnotatedPtr { type_id, ptr }
    }

    fn dealloc(&self) {
        let ptr_to_dealloc = self.ptr.cast::<u8>().as_ptr();
        if ptr_to_dealloc != &mut ZERO_SIZED_TYPE_REPRESENTATIVE as * mut u8 {
            unsafe {
                Jemalloc.dealloc
                    (self.ptr.cast::<u8>().as_ptr(),
                    get_layout_for_type_id_assert_present(self.type_id));
            }
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

#[derive(Copy, Clone, PartialEq, Eq)]
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

    fn get_inner_raw_version_ptr(&self) -> NonNull<()>
    {
        self.0.ptr
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
        assert!(std::mem::align_of::<TVarVersionInner<GuardedType>>() >= 2);
        DynamicInnerTVarPtr {
            0: TypeIdAnnotatedPtr::alloc(TVarVersionInner {
                timestamp: NON_CANON_TIMESTAMP.clone(),
                next_ptr: null(),
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

    // Gets the timestamp on a pointee. This version just naively returns the
    // value that is present.
    fn get_pointee_timestamp_val(self) -> u64 {
        let type_erased_inner_version_ptr =
            self.0.ptr.cast::<TVarVersionInner<()>>();
        unsafe {
            // This must be an acquire load. We may be loading a version that
            // does not happen-before this transaction, in which case we must
            // acquire to be able to inspect the timestamp that it points to.
            type_erased_inner_version_ptr.as_ref().timestamp.0.load(Acquire)
        }
    }

    // This function, on the other hand, expects that we are fetching the
    // timestamp of a canonical version. It will assert if the version is
    // non-canonical and will loop until the version is filled in if it has yet
    // to be committed.
    fn get_pointee_timestamp_val_expect_canon(self) -> u64 {
        loop {
            let version_val = self.get_pointee_timestamp_val();
            assert!(version_val != TXN_COUNTER_NON_CANON);
            if version_val == TXN_COUNTER_UPDATING_VAL {
                spin_loop_hint();
            } else {
                return version_val;
            }
        }
    }
}

#[derive(PartialEq, Eq)]
struct TVarImmVersion<GuardedType> {
    type_erased_inner: DynamicInnerTVarPtr,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType> Clone for TVarImmVersion<GuardedType> {
    fn clone(&self) -> Self {
        TVarImmVersion {
            type_erased_inner: self.type_erased_inner.clone(),
            phantom: PhantomData
        }
    }
}
impl<GuardedType> Copy for TVarImmVersion<GuardedType> { }

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

// A version of Clone specifically for use in cloning versions of a tvar. This
// is used because there are circumstances in which a type should not be
// clonable in the traditional sense, but should be able to be cloned within a
// tvar.
pub trait TVarVersionClone {
    fn tvar_version_clone(&self) -> Self;
}

impl<ClonableType: Clone> TVarVersionClone for ClonableType {
    fn tvar_version_clone(&self) -> Self {
        self.clone()
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

impl<GuardedType: 'static> TVarShadowVersion<GuardedType> {

    fn new(guarded: GuardedType) -> TVarShadowVersion<GuardedType> {
        TVarShadowVersion {
            type_erased_inner: DynamicInnerTVarPtr::alloc(guarded),
            phantom: PhantomData
        }
    }

    fn get_dyn_inner_version_ptr(&self) -> DynamicInnerTVarPtr {
        self.type_erased_inner
    }

}

impl<GuardedType: TVarVersionClone + 'static> TVarShadowVersion<GuardedType> {

    fn clone_from_imm_version
        (imm_version: TVarImmVersion<GuardedType>) ->
            TVarShadowVersion<GuardedType>
    {
        Self::new
            (imm_version
                .get_dyn_inner_version_ptr()
                .get_guarded_ref::<GuardedType>()
                .tvar_version_clone())
    }
}

#[derive(Copy, Clone)]
struct VersionedTVarTypeErasedRef {
    tvar_ref: &'static VersionedTVarTypeErased,
    version: u64
}

struct VersionedTVarRef<GuardedType> {
    type_erased_ref: VersionedTVarTypeErasedRef,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType> Clone for VersionedTVarRef<GuardedType> {
    fn clone(&self) -> Self {
        VersionedTVarRef {
            type_erased_ref: self.type_erased_ref.clone(),
            phantom: PhantomData
        }
    }
}
impl<GuardedType> Copy for VersionedTVarRef<GuardedType> { }

struct CanonPtrAndWriteReserved {
    canon_ptr: NonNull<()>,
    write_reserved: bool
}

impl CanonPtrAndWriteReserved {
    fn new<Pointee>(pointer: * mut Pointee, write_reserved: bool)
        -> CanonPtrAndWriteReserved
    {
        let canon_ptr_as_int = pointer as u64;
        // Assert that the bottom bit of the pointer is 0. If this isn't the
        // case, we can't pack it.
        assert!(canon_ptr_as_int & 1 == 0);

        CanonPtrAndWriteReserved {
            canon_ptr: NonNull::new(pointer as * mut ()).unwrap(),
            write_reserved
        }
    }

    fn pack(&self) -> u64 {
        let canon_ptr_as_int = self.canon_ptr.as_ptr() as u64;
        // Assert that the bottom bit of the pointer is 0.
        assert!(canon_ptr_as_int & 1 == 0);
        canon_ptr_as_int | if self.write_reserved { 1 } else { 0 }
    }

    fn unpack(packed: u64) -> CanonPtrAndWriteReserved {
        let write_reserved = (packed & 1) == 1;
        let canon_ptr = ((packed >> 1) << 1) as * mut ();
        CanonPtrAndWriteReserved::new(canon_ptr, write_reserved)
    }
}

struct VersionedTVarTypeErased {
    // TVars are never deallocated in the standard sense. They can, however, be
    // reused. This means that a pointer to a tvar can always be dereferenced,
    // but the pointer may or may not be to the same object. This version is
    // incremented upon every "deallocation" of a tvar and is placed upon each
    // VersionedTVarPointer, allowing us to check for liveness without doing
    // reference counting.
    version: AtomicU64,
    packed_canon_ptr: AtomicU64,
    stale_version_list: AtomicPtr<()>,
    type_id: TypeId,
    free_list_next: AtomicPtr<VersionedTVarTypeErased>
}

impl VersionedTVarTypeErased {
    fn fetch_and_unpack_canon_ptr
        (&self, ordering: Ordering) -> CanonPtrAndWriteReserved
    {
        let packed_canon_ptr = self.packed_canon_ptr.load(ordering);
        CanonPtrAndWriteReserved::unpack(packed_canon_ptr)
    }

    fn has_canon_version_ptr(&self, candidate: NonNull<()>) -> bool {
        self.fetch_and_unpack_canon_ptr(Relaxed).canon_ptr == candidate
    }

    fn borrow(&'static self) -> VersionedTVarTypeErasedRef {
        VersionedTVarTypeErasedRef {
            tvar_ref: &self,
            // XXX: Is acquire necessary here? I think I could get away with
            // Relaxed.
            version: self.version.load(Acquire)
        }
    }

    // This function gets the current canon version or, if a special status code
    // pointer is in the canon pointer, the status.
    fn get_current_canon_version_and_reserved
        (&self, expected_tvar_version: Option<u64>)
        -> Result<CanonPtrAndWriteReserved, TxnErrStatus>
    {
        let initial_version = self.version.load(Acquire);
        let canon_ptr_and_write_reserved =
            self.fetch_and_unpack_canon_ptr(Acquire);
        let second_version = self.version.load(Acquire);

        if initial_version != second_version {
            return Err(TxnErrStatus { });
        }
        match expected_tvar_version {
            Some(expected_version_num) => {
                if expected_version_num != initial_version {
                    return Err(TxnErrStatus { });
                }
            },
            None => { }
        }
        Ok(canon_ptr_and_write_reserved)
    }

    fn get_current_canon_version(&self, expected_version: Option<u64>)
        -> Result<DynamicInnerTVarPtr, TxnErrStatus>
    {
        let canon_ptr_and_write_reserved =
            self.get_current_canon_version_and_reserved(expected_version)?;

        let type_id_annot_ptr =
            TypeIdAnnotatedPtr {
                ptr: canon_ptr_and_write_reserved.canon_ptr,
                type_id: self.type_id };
        let dynamic_inner_version_ptr = DynamicInnerTVarPtr {
            0: type_id_annot_ptr
        };
        return Ok(dynamic_inner_version_ptr)
    }

    fn clear_write_reservation(&self) {
        let all_bits_except_bottom: u64 = !1;
        self.packed_canon_ptr.fetch_and(all_bits_except_bottom, Relaxed);
    }
}

pub struct VersionedTVar<GuardedType> {
    inner_type_erased: VersionedTVarTypeErased,
    phantom: PhantomData<GuardedType>
}

unsafe impl<GuardedType> Send for VersionedTVar<GuardedType> { }

impl<GuardedType : TVarVersionClone + 'static> VersionedTVar<GuardedType> {
    pub fn new(inner_val: GuardedType) -> VersionedTVar<GuardedType> {
        // To use the same flow of functions that a regular update to the TVar
        // uses, just create a shadow version and turn it into a TVarImmVersion.

        let init_shadow_version = TVarShadowVersion::new(inner_val);

        let init_type_erased_inner = init_shadow_version.type_erased_inner;

        let mut inner_version_ptr =
            init_type_erased_inner
                .downcast_to_inner_version_having_guarded::<GuardedType>();

        unsafe {
            inner_version_ptr.as_mut().timestamp = INIT_TIMESTAMP.clone();
        }

        let init_dynamic_inner_version_ptr = init_type_erased_inner;
        let canon_ptr_and_write_reserved =
            CanonPtrAndWriteReserved::new
                (init_dynamic_inner_version_ptr.0.ptr.as_ptr(), false);
        let result: VersionedTVar<GuardedType> =
            VersionedTVar {
                inner_type_erased: VersionedTVarTypeErased {
                    packed_canon_ptr:
                        AtomicU64::new(canon_ptr_and_write_reserved.pack()),
                    stale_version_list: AtomicPtr::new(null_mut()),
                    type_id: init_dynamic_inner_version_ptr.0.type_id,
                    version: AtomicU64::new(0),
                    free_list_next: AtomicPtr::new(null_mut())
                },
                phantom: PhantomData
            };
        result
    }

    // This function spins until it gets the current canon version.
    fn get_current_canon_version(&self, expected_version: Option<u64>)
        -> Result<TVarImmVersion<GuardedType>, TxnErrStatus>
    {
        let type_erased_inner =
            self.inner_type_erased.get_current_canon_version(expected_version)?;
        Ok(TVarImmVersion { type_erased_inner, phantom: PhantomData })
    }

    fn borrow(&'static self) -> VersionedTVarRef<GuardedType> {
        VersionedTVarRef {
            type_erased_ref: self.inner_type_erased.borrow(),
            phantom: PhantomData
        }
    }
}

struct VersionedTVarAllocator {
    unclaimed_tvars: AtomicPtr<VersionedTVarTypeErased>
}

unsafe impl Send for VersionedTVarAllocator { }
unsafe impl Sync for VersionedTVarAllocator { }

impl VersionedTVarAllocator {

    // This is for use in allocating tvars that should have a certain object
    // identity for less than the static lifetime.
    fn get_new_tvar_ref<GuardedType: TVarVersionClone + 'static>
        (&mut self, contents: GuardedType)
            -> VersionedTVarRef<GuardedType>
    {
        let mut unclaimed_head_ptr = self.unclaimed_tvars.load(Acquire);
        let type_erased_versioned_tvar_ref =
            loop {
                // If the unclaimed head is null, there are no unclaimed tvars
                // to be had and we have to allocate a new one.
                if unclaimed_head_ptr == null_mut() {
                    // We leak this because we will never delete the tvar. Maybe
                    // we will attempt to reclaim tvar memory at some point, but
                    // doing that correctly seems hard.
                    let new_leaked_tvar =
                        Box::leak(Box::new(VersionedTVar::new(contents)));
                    break &(*new_leaked_tvar).inner_type_erased;
                } else {
                    let new_head_ptr =
                        unsafe {
                            (*unclaimed_head_ptr).free_list_next.load(Relaxed)
                        };
                    let cmp_ex_result =
                        self.unclaimed_tvars.compare_exchange_weak
                            (unclaimed_head_ptr,
                             new_head_ptr,
                             Acquire,
                             Acquire);
                    match cmp_ex_result {
                        Ok(_) => {
                            break unsafe { &*unclaimed_head_ptr };
                        }
                        Err(found_pointer) => {
                            unclaimed_head_ptr = found_pointer;
                            spin_loop_hint();
                            continue;
                        }
                    }
                }
            };
        VersionedTVarRef {
            type_erased_ref: VersionedTVarTypeErasedRef {
                tvar_ref: type_erased_versioned_tvar_ref,
                version: type_erased_versioned_tvar_ref.version.load(Relaxed)
            },
            phantom: PhantomData
        }
    }
}

static mut TVAR_ALLOCATOR: VersionedTVarAllocator =
    VersionedTVarAllocator { unclaimed_tvars: AtomicPtr::new(null_mut()) };

// It's often quite unfortunate to have to copy a whole structure if only a
// small portion of it changed. This TVarCell provides a way for us to modify a
// field of a large TVar without having to copy the whole thing.
#[derive(Clone, Copy)]
pub struct TVarCell<GuardedType: TVarVersionClone + 'static> {
    cell_payload: VersionedTVarRef<GuardedType>
}

impl<GuardedType: TVarVersionClone + 'static> TVarCell<GuardedType> {
    pub fn new(val: GuardedType) -> TVarCell<GuardedType> {
        TVarCell {
            cell_payload: unsafe { TVAR_ALLOCATOR.get_new_tvar_ref(val) }
        }
    }
}

struct CapturedTVarTypeErased {
    captured_version_ptr: NonNull<()>,
    shadow_copy_ptr: Option<NonNull<()>>,
    type_id: TypeId,
    // The CapturedTVar acts like a RefCell, but not exactly, because we want to
    // free up the hash map for insertion while borrowing individual cells.
    mut_borrowed: Cell<bool>,
    num_shared_borrows: Cell<u64>,
    index: CapturedTVarIndex
}

impl CapturedTVarTypeErased {

    fn new<GuardedType: 'static> (captured_version: TVarImmVersion<GuardedType>)
        -> CapturedTVarTypeErased
    {
        let dyn_inner_version_ptr =
            captured_version.get_dyn_inner_version_ptr();
        CapturedTVarTypeErased {
            captured_version_ptr: dyn_inner_version_ptr.0.ptr,
            type_id: dyn_inner_version_ptr.0.type_id,
            shadow_copy_ptr: None,
            mut_borrowed: Cell::new(false),
            num_shared_borrows: Cell::new(0),
            index: INVALID_CAPTURED_TVAR_INDEX
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

    fn get_shadow_copy_create_if_not_present
        <GuardedType: TVarVersionClone + 'static>(&mut self)
            -> TVarShadowVersion<GuardedType>
    {
        match self.get_optional_shadow_copy() {
            Some(already_present_shadow_copy) => already_present_shadow_copy,
            None => {
                let captured_version =
                    self.get_captured_version::<GuardedType>();
                let new_shadow_version =
                    TVarShadowVersion::new
                        (captured_version
                            .get_dyn_inner_version_ptr()
                            .get_guarded_ref::<GuardedType>()
                            .tvar_version_clone());
                self.shadow_copy_ptr =
                    Some(new_shadow_version.type_erased_inner.0.ptr);
                new_shadow_version
            }
        }
    }

    fn discard_shadow_state(&mut self) {
        let previous_shadow_copy = self.shadow_copy_ptr.take();
        match previous_shadow_copy {
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

    fn borrow<'tvar_ref, GuardedType: 'static>(&self)
        -> CapturedTVarRef<'tvar_ref, GuardedType>
    {
        if self.mut_borrowed.get() {
            panic!
                ("Attempted to borrow a CapturedTVar as mutable and \
                 shared simultaneously within one thread.")
        }
        self.num_shared_borrows.set(self.num_shared_borrows.get() + 1);
        let opt_shadow_copy_ref =
            self.get_optional_shadow_copy::<GuardedType>();
        let tvar_ref =
            match opt_shadow_copy_ref {
                Some(present_shadow_copy) =>
                    present_shadow_copy
                        .get_dyn_inner_version_ptr()
                        .get_guarded_ref(),
                None =>
                self.get_captured_version::<GuardedType>()
                    .get_dyn_inner_version_ptr()
                    .get_guarded_ref()
            };
        CapturedTVarRef {
            inner_ref: tvar_ref,
            index: self.index
        }
    }

    fn borrow_mut<'tvar_mut, GuardedType: 'static + TVarVersionClone>
        (&mut self)
            -> CapturedTVarMut<'tvar_mut, GuardedType>
    {
        if (self.mut_borrowed.get()) || (self.num_shared_borrows.get() != 0) {
            panic!
                ("Attempted to borrow a CapturedTVar as mutable and \
                 shared simultaneously within one thread.")
        }
        self.mut_borrowed.set(true);
        let tvar_mut =
            self.get_shadow_copy_create_if_not_present::<GuardedType>()
                .get_dyn_inner_version_ptr()
                .get_guarded_mut();
        CapturedTVarMut {
            inner_mut: tvar_mut,
            index: self.index
        }
    }
}

#[derive(Clone, Copy)]
struct CapturedTVarIndex(usize);

const INVALID_CAPTURED_TVAR_INDEX: CapturedTVarIndex =
    CapturedTVarIndex { 0: std::usize::MAX };

pub struct CapturedTVarRef<'a, GuardedType> {
    index: CapturedTVarIndex,
    inner_ref: &'a GuardedType
}

impl<'a, GuardedType> Borrow<GuardedType> for CapturedTVarRef<'a, GuardedType> {
    fn borrow(&self) -> &GuardedType {
        self.inner_ref
    }
}

impl<'a, GuardedType> Deref for CapturedTVarRef<'a, GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        self.inner_ref
    }
}

impl<'a, GuardedType> Drop for CapturedTVarRef<'a, GuardedType> {
    fn drop(&mut self) {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state_key|{
            let per_thread_state_ref =
                per_thread_txn_state_key.borrow();

            let captured_tvar_mut =
                &mut per_thread_state_ref
                    .captured_tvar_cache
                    .get_index(self.index.0)
                    .unwrap()
                    .1
                    .borrow_mut();
            assert!(!captured_tvar_mut.mut_borrowed.get());
            let num_shared_borrows =
                captured_tvar_mut.num_shared_borrows.get();
            assert!(num_shared_borrows > 0);
            captured_tvar_mut.num_shared_borrows.set(num_shared_borrows - 1);
        });
    }
}

pub struct CapturedTVarMut<'a, GuardedType> {
    index: CapturedTVarIndex,
    inner_mut: &'a mut GuardedType
}

impl<'a, GuardedType> Borrow<GuardedType> for CapturedTVarMut<'a, GuardedType> {
    fn borrow(&self) -> &GuardedType {
        return self.inner_mut;
    }
}

impl<'a, GuardedType>
BorrowMut<GuardedType> for CapturedTVarMut<'a, GuardedType> {
    fn borrow_mut(&mut self) -> &mut GuardedType {
        return self.inner_mut;
    }
}

impl<'a, GuardedType> Deref for CapturedTVarMut<'a, GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        self.inner_mut
    }
}

impl<'a, GuardedType> DerefMut for CapturedTVarMut<'a, GuardedType> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner_mut
    }
}

impl<'a, GuardedType> Drop for CapturedTVarMut<'a, GuardedType> {
    fn drop(&mut self) {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state_key|{
            let per_thread_state_ref =
                per_thread_txn_state_key.borrow();

            let captured_tvar_mut =
                &mut per_thread_state_ref
                    .captured_tvar_cache
                    .get_index(self.index.0)
                    .unwrap()
                    .1
                    .borrow_mut();
            assert!(captured_tvar_mut.num_shared_borrows.get() == 0);
            assert!(captured_tvar_mut.mut_borrowed.get());
            captured_tvar_mut.mut_borrowed.set(false);
        });
    }
}

pub struct CapturedTVarCacheKey<'key, GuardedType> {
    cache_index: CapturedTVarIndex,
    phantom: PhantomData<GuardedType>,
    phantom_lifetime: PhantomData<&'key ()>
}

impl<'key, GuardedType: TVarVersionClone + 'static>
CapturedTVarCacheKey<'key, GuardedType> {
    fn with_captured_tvar_ref_cell
        <ReturnType, TVarFn: FnOnce(&RefCell<CapturedTVarTypeErased>) -> ReturnType>
        (&self, tvar_fn: TVarFn) -> ReturnType
    {
        PER_THREAD_TXN_STATE.with(|per_thread_state_key| {
            let per_thread_state_ref = per_thread_state_key.borrow();
            let captured_tvar_ref_cell =
                per_thread_state_ref
                    .captured_tvar_cache
                    .get_index(self.cache_index.0)
                    .unwrap()
                    .1;
                tvar_fn(captured_tvar_ref_cell)
        })
    }

    pub fn get_captured_tvar_ref<'txn: 'tvar_ref, 'tvar_ref>(&self)
            -> CapturedTVarRef<'tvar_ref, GuardedType>
        where 'key: 'tvar_ref
    {
        self.with_captured_tvar_ref_cell(|captured_tvar|{
            let captured_tvar_ref = captured_tvar.borrow();
            (*captured_tvar_ref).borrow()
        })
    }

    pub fn get_captured_tvar_mut<'tvar_mut>(&self)
            -> CapturedTVarMut<'tvar_mut, GuardedType>
        where 'key: 'tvar_mut
    {
        PER_THREAD_TXN_STATE.with(|per_thread_state_key|{
            let mut per_thread_state_mut = per_thread_state_key.borrow_mut();
            *per_thread_state_mut.is_write_txn.borrow_mut() = true;
        });

        self.with_captured_tvar_ref_cell(|captured_tvar|{
            let mut captured_tvar_mut = captured_tvar.borrow_mut();
            (*captured_tvar_mut).borrow_mut()
        })
    }

    pub fn discard_changes(&self) {
        self.with_captured_tvar_ref_cell(|captured_tvar|{
            (*captured_tvar.borrow_mut()).discard_shadow_state();
        })

    }
}

lazy_static! {
    static ref WRITE_TXN_TIME: AtomicU64 = AtomicU64::new(0);
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
        IndexMap<NonNull<VersionedTVarTypeErased>,
                 RefCell<CapturedTVarTypeErased>>,
    is_write_txn: RefCell<bool>,
    // Writes at and before this version happened-before this txn, and thus are
    // guaranteed to be consistent. Writes after it, on the other hand, can be
    // skewed, and require a new acquire of the counter.
    txn_version_acquisition_threshold: u64,
}

impl PerThreadTransactionState {
    fn reset_txn_state(&mut self, drop_shadow_versions: bool) {
        if drop_shadow_versions {
            for (_, captured_tvar_cell) in
                self.captured_tvar_cache.drain(..)
            {
                captured_tvar_cell.borrow_mut().discard_shadow_state();
            }
        } else {
            self.captured_tvar_cache.clear();
        }
        *self.is_write_txn.borrow_mut() = false
    }
}

thread_local! {
    static PER_THREAD_TXN_STATE: RefCell<PerThreadTransactionState> =
        RefCell::new(
            PerThreadTransactionState {
                captured_tvar_cache: IndexMap::new(),
                txn_version_acquisition_threshold: 0,
                is_write_txn: RefCell::new(false)
            });
}

// A transaction error. This is used to roll back the transaction on failure.
#[derive(Debug)]
pub struct TxnErrStatus { }

// This enum is to help in VersionedTransaction's hit_tvar_cache method.
enum TVarIndexResult<GuardedType> {
    KnownFreshIndex(usize),
    MaybeNotFreshContext(TVarImmVersion<GuardedType>, u64)
}

// This struct actually contains little transaction data. It acts as a gateway
// to the actual transaction state, which is stored in thread local storage.
// Because the per-thread transaction state cannot be accessed unless one is in
// a transaction, the pattern of allowed accesses is actually more intuitive if
// a user acts as if the transaction state were stored within the
// VersionedTransaction.
pub struct VersionedTransaction {
    txn_succeeded: bool
}

impl Drop for VersionedTransaction {
    fn drop(& mut self) {
        // When a transaction has finished, we should always make sure that the
        // per-thread state is cleared.
        self.with_per_thread_txn_state_mut(|per_thread_state_mut|{
            let should_drop_shadow_versions = !self.txn_succeeded;
            per_thread_state_mut.reset_txn_state(should_drop_shadow_versions);
        })
    }
}

impl VersionedTransaction {

    fn with_per_thread_txn_state_ref
        <ReturnType,
         PerThreadTxnFn: FnOnce(&PerThreadTransactionState) -> ReturnType>
    (&self, per_thread_txn_fn: PerThreadTxnFn) -> ReturnType
    {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state_key|{
            per_thread_txn_fn(&*per_thread_txn_state_key.borrow())
        })
    }

    fn with_per_thread_txn_state_mut
        <ReturnType,
         PerThreadTxnFn: FnOnce(&mut PerThreadTransactionState) -> ReturnType>
    (&self, per_thread_txn_fn: PerThreadTxnFn) -> ReturnType
    {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state_key|{
            per_thread_txn_fn(&mut *per_thread_txn_state_key.borrow_mut())
        })
    }

    fn make_transaction_error(&self) -> TxnErrStatus {
        TxnErrStatus { }
    }

    fn perform_txn
        <UserResult,
         ThisTxnFn:
            Fn(&mut VersionedTransaction) -> Result<UserResult, TxnErrStatus>
        >
        (&mut self, txn_fn: ThisTxnFn) -> UserResult
    {
        loop {
            self.with_per_thread_txn_state_mut(|per_thread_state_mut| {
                per_thread_state_mut.reset_txn_state
                    (true/*should_drop_shadow_versions*/);
            });
            let txn_fn_result = txn_fn(self);
            let user_result =
                match txn_fn_result {
                    Ok(ok_result) => {
                        ok_result
                    },
                    Err(_) => {
                        continue;
                    }
                };

            // We have successfully completed the user's function without
            // hitting any errors. Now we have to perform the commit action.
            // If this transaction was a read transaction, nothing more to do.
            let is_write_txn =
                self.with_per_thread_txn_state_ref
                    (|per_thread_state_ref| {
                         *per_thread_state_ref.is_write_txn.borrow()
                    });
            if is_write_txn {
                let commit_succeeded =
                    PER_THREAD_TXN_STATE.with(|per_thread_state_key| {
                        self.perform_write_txn_commit
                            (&mut *per_thread_state_key.borrow_mut())
                    });
                if !commit_succeeded {
                    continue;
                }
            } else {
                self.txn_succeeded = true;
            }

            return user_result;
        }
    }

    // For a write transaction, commit the changes. Returns whether or not the
    // commit succeeded.
    fn perform_write_txn_commit
        (&mut self, txn_state: &mut PerThreadTransactionState)
        -> bool
    {
        let captured_tvar_cache_mut = &mut txn_state.captured_tvar_cache;

        // If this is a write transaction, we must swap the special
        // write-reserve value in for all tvars considered. If we encounter any
        // unexpected values along the way, another write must have slipped by
        // us and we must abort.
        captured_tvar_cache_mut.sort_keys();
        let mut reserve_for_write_iter =
            captured_tvar_cache_mut.iter().enumerate();

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
                captured_tvar.borrow().captured_version_ptr.as_ptr();
            let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
            let unreserved_canon_ptr =
                CanonPtrAndWriteReserved::new(expected_canon_ptr, false);
            let reserved_canon_ptr =
                CanonPtrAndWriteReserved::new(expected_canon_ptr, true);
            'cmp_ex_loop: loop {
                // We are swapping in a marker pointer with no contents,
                // and whether we fail or succeed, we care only about
                // the pointer values, not any associated data. I think
                // we can get away with Relaxed ordering here.
                let cmp_ex_result =
                    tvar_ref.packed_canon_ptr.compare_exchange_weak
                        (unreserved_canon_ptr.pack(),
                         reserved_canon_ptr.pack(),
                         Relaxed,
                         Relaxed);
                match cmp_ex_result {
                    Ok(_) => {
                        // Remember, this is successful swap *length*, so
                        // we want 1 plus the idx.
                        successful_swap_length = idx + 1;
                        continue 'reserve_commit;
                    },
                    Err(found_packed_ptr) => {
                        let unpacked_found_ptr =
                            CanonPtrAndWriteReserved::unpack(found_packed_ptr);
                        // If the found pointer did not match, another commit
                        // has beaten us to the punch.
                        if unpacked_found_ptr.canon_ptr !=
                             unreserved_canon_ptr.canon_ptr {
                            saw_conflict = true;
                            break 'reserve_commit;
                        }
                        // If the pointers were equal, but the one in the tvar
                        // has the write reserved flag, another transaction has
                        // reserved the tvar for writing. Spin til it is
                        // available. If that is not the case, we have
                        // encountered a spurious failure and should retry,
                        // which is the same behavior.
                        //
                        // XXX: Should this spin_loop_hint be triggered on
                        // spurious failures?
                        spin_loop_hint();
                        continue 'cmp_ex_loop;
                    }
                }
            }
        }

        // If we saw a swap failure, we have to put everything back as
        // we found it and roll back the transaction.
        if saw_conflict {
            for (idx, (tvar_ptr, _)) in
                captured_tvar_cache_mut.iter().enumerate()
            {
                if idx == successful_swap_length {
                    return false;
                }
                let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
                tvar_ref.clear_write_reservation();
            }
            // The above loop should always return false eventually, so this
            // should be unreachable.
            unreachable!();
        }

        // We have reached the point where the write transaction is
        // guaranteed to succeed.
        self.txn_succeeded = true;

        // Create a timestamp for this transaction, initially with the
        // UPDATING value. We will fill this in with the actual value once
        // all txn values have been swapped in.
        let this_txn_timestamp = TxnTimestamp::new_updating();

        // Now we must update all of the tvars we have touched. For those
        // that we did not modify, we must clear the write reserved bit. For
        // those that we did, we must swap in our shadow version upgraded to
        // a new immutable version.
        for (tvar_ptr, captured_tvar) in captured_tvar_cache_mut.iter() {
            let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
            match captured_tvar.borrow().shadow_copy_ptr {
                Some(inner_shadow_copy_ptr) => {
                    let raw_shadow_copy_ptr = inner_shadow_copy_ptr.as_ptr();
                    let ptr_to_shadow_copy_header =
                        raw_shadow_copy_ptr as * mut TVarVersionInner<()>;
                    unsafe {
                        (*ptr_to_shadow_copy_header).timestamp =
                            this_txn_timestamp.clone();
                    }

                    // This is our first time publishing this shadow version
                    // to other threads, so we must use a release ordering.
                    tvar_ref
                        .packed_canon_ptr
                        .store
                            (CanonPtrAndWriteReserved::new
                                (raw_shadow_copy_ptr, false).pack(),
                             Release)
                },
                None => tvar_ref.clear_write_reservation()
            }
        }

        // At this point, we have updated all of the tvars with our new
        // versions, but none of those versions have a canonical time. Now,
        // we must fetch the canonical time for this transaction. We perform
        // a fetch-add with a release ordering. This means that any
        // Acquire read seeing at least the number we move the counter to
        // must also happen-after all of the swaps we just performed.

        let old_counter_value = TXN_WRITE_TIME.fetch_add(1, Release);
        let this_txn_time_number = old_counter_value + 1;

        // This marks all of the versions we have swapped in with the
        // current timestamp, completing the commit phase of this
        // transaction.
        this_txn_timestamp.0.store(this_txn_time_number, Release);

        // Now that we have completed the commit, we still have to
        // perform some cleanup actions. For all captured tvars where we
        // swapped in a new value, we must place the old canonical version
        // on the stale list.
        for (tvar_ptr, captured_tvar) in captured_tvar_cache_mut.drain(..) {
            // There's nothing to do for captured tvars which did not have a
            // shadow copy (ie, were not writes.)
            if captured_tvar.borrow().shadow_copy_ptr.is_none() {
                continue;
            }
            // For writes, place the old canon version onto the stale
            // version list.
            let old_canon =
                captured_tvar.borrow().captured_version_ptr.as_ptr();
            let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
            std::mem::forget(captured_tvar);
            loop {
                let old_stale_list_head =
                    tvar_ref.stale_version_list.load(Relaxed);
                let old_canon_as_unknown_version_inner =
                    old_canon as * mut TVarVersionInner<()>;
                unsafe {
                    (*old_canon_as_unknown_version_inner).next_ptr =
                        old_stale_list_head;
                }
                // Try to place the old canon at the front of the stale
                // list. We use Release because we want our store for the
                // next pointer to show up to any future Acquirers. The
                // failure mode for the load on acquire can be weak,
                // however, because we don't even look at the pointer we
                // get back when we fail.
                let cmp_ex_result =
                    tvar_ref
                        .stale_version_list
                        .compare_exchange_weak
                            (old_stale_list_head,
                                old_canon,
                                Release,
                                Relaxed);
                match cmp_ex_result {
                    Ok(_) => break,
                    Err(_) => continue
                }
            }
        }

        return true;
    }

    pub fn start_txn
        <UserResult,
         ThisTxnFn:
            Fn(&mut VersionedTransaction) -> Result<UserResult, TxnErrStatus>
        >
        (txn_fn: ThisTxnFn) -> UserResult
    {
        // A top-level transaction will borrow from the per-thread transaction
        // state and assume that no other transaction has checked it out.
        let mut txn = VersionedTransaction { txn_succeeded: false };
        txn.perform_txn(txn_fn)
    }

    // This function checks that all tvars captured by the current transaction
    // are still up to date. Uses Result with an uninteresting Ok as the return
    // address so it can use the ? operator to propagate the transaction error
    // on failure. The transaction error contains the earliest transaction
    // number that may need to be restarted.
    fn check_all_captured_current(&self) -> Result<(), TxnErrStatus>
    {
        self.with_per_thread_txn_state_ref(|per_thread_txn_state_ref| {
            for (type_erased_tvar_ptr, captured_tvar) in
                    per_thread_txn_state_ref.captured_tvar_cache.iter()
            {
                let type_erased_tvar = unsafe { type_erased_tvar_ptr.as_ref() };
                if !type_erased_tvar.has_canon_version_ptr
                    (captured_tvar.borrow().captured_version_ptr)
                {
                    return Err(self.make_transaction_error());
                }
            }
            Ok(())
        })
    }

    // To reduce the amount of code used, capturing a tvar creates a deferred
    // tvar capture and immediately completes it.
    pub fn capture_tvar<GuardedType: TVarVersionClone + 'static>
        (&self, tvar: &'static VersionedTVar<GuardedType>)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        self.capture_tvar_with_optional_expected_version
            (&tvar.inner_type_erased, None)
    }

    pub fn capture_tvar_cell<GuardedType: TVarVersionClone + 'static>
        (&self, tvar_cell: &TVarCell<GuardedType>)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        let inner_tvar_ref = tvar_cell.cell_payload.type_erased_ref;
        self.capture_tvar_with_optional_expected_version
            (inner_tvar_ref.tvar_ref, Some(inner_tvar_ref.version))
    }

    fn capture_tvar_with_optional_expected_version
        <GuardedType: TVarVersionClone + 'static>
        (&self,
         tvar: &'static VersionedTVarTypeErased,
         expected_version: Option<u64>)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        let cache_key = NonNull::from(tvar);

        let version_acquisition_threshold =
            self.with_per_thread_txn_state_ref(|per_thread_state_ref|{
                per_thread_state_ref.txn_version_acquisition_threshold
            });

        let index_lookup_result = self.with_per_thread_txn_state_mut
            (|per_thread_state_mut| {
                let captured_tvar_entry =
                    per_thread_state_mut.captured_tvar_cache.entry(cache_key);
                let tvar_index_result: TVarIndexResult<GuardedType> =
                    match captured_tvar_entry
                    {
                    // If the entry is already there, return the known index.
                    Occupied(present_entry) => {
                        TVarIndexResult::KnownFreshIndex
                            (present_entry.index())
                    },
                    Vacant(vacant_entry) => {
                        let versioned_tvar_type_erased =
                            unsafe { cache_key.as_ref() };
                        let current_canon =
                            versioned_tvar_type_erased
                                .get_current_canon_version(expected_version)?;
                        let current_canon_tvar_imm_version = TVarImmVersion {
                            type_erased_inner: current_canon,
                            phantom: PhantomData
                        };
                        let mut captured_tvar =
                            CapturedTVarTypeErased::new
                                (current_canon_tvar_imm_version);
                        let new_entry_idx = vacant_entry.index();
                        captured_tvar.index =
                            CapturedTVarIndex { 0: new_entry_idx };

                        // The captured tvar we inserted may have a timestamp
                        // that did not happen before the current transaction
                        // start. If that's the case, we need to check that all
                        // captured entries (except the just captured one) are
                        // still current.
                        let dyn_inner_version_ptr =
                            captured_tvar
                                .get_captured_version::<GuardedType>()
                                .get_dyn_inner_version_ptr();
                        let pointee_timestamp =
                            dyn_inner_version_ptr
                                .get_pointee_timestamp_val_expect_canon();
                        // If the timestamp is earlier than the acquisition
                        // threshold, we are in good shape and can insert the
                        // entry into the map.
                        if pointee_timestamp <= version_acquisition_threshold {
                            vacant_entry.insert(RefCell::new(captured_tvar));
                            TVarIndexResult::KnownFreshIndex(new_entry_idx)
                        } else {
                            // Otherwise, it is not known-fresh. Return the
                            // information that we have to try to get the index
                            // again after we fast-forward our state.
                            TVarIndexResult::MaybeNotFreshContext
                                (current_canon_tvar_imm_version,
                                 pointee_timestamp)
                        }
                    }
                };
                Ok(tvar_index_result)
            })?;

        let final_index =
            match index_lookup_result {
                TVarIndexResult::KnownFreshIndex(index) => {
                    index
                },
                TVarIndexResult::MaybeNotFreshContext
                    (current_canon, pointee_timestamp) =>
                {
                    // If we reached this point, the timestamp was not at or
                    // before the acquisition threshold. We must try to update
                    // our transaction timestamp and check all captured
                    // versions to ensure that they are current.
                    let new_acquisition_threshold =
                        TXN_WRITE_TIME.load(Acquire);
                    self.with_per_thread_txn_state_mut(
                        |per_thread_txn_state_mut| {
                            per_thread_txn_state_mut
                                .txn_version_acquisition_threshold =
                                    new_acquisition_threshold;
                        });
                    self.check_all_captured_current()?;
                    assert!(new_acquisition_threshold >= pointee_timestamp);

                    self.with_per_thread_txn_state_mut
                        (|per_thread_txn_state_mut| {
                        let (index, previous_value_expect_none) =
                            per_thread_txn_state_mut
                                .captured_tvar_cache
                                .insert_full
                                    (cache_key,
                                     RefCell::new
                                        (CapturedTVarTypeErased::new
                                            (current_canon)));
                        assert!(previous_value_expect_none.is_none());
                        per_thread_txn_state_mut
                            .captured_tvar_cache
                            .get_index(index)
                            .unwrap()
                            .1
                            .borrow_mut()
                            .index = CapturedTVarIndex { 0: index };
                        index
                    })
                }
            };
        Ok(CapturedTVarCacheKey {
            cache_index: CapturedTVarIndex(final_index),
            phantom: PhantomData,
            phantom_lifetime: PhantomData
        })
    }
}

#[cfg(test)]
mod tests {

    use std::thread;
    use std::thread::JoinHandle;
    use std::sync::Mutex;
    use std::sync::Condvar;
    use super::*;

    mod test1_state {
        use super::*;
        lazy_static! {
            pub static ref TVAR_INT1: VersionedTVar<u64> =
                VersionedTVar::new(5);
            pub static ref TVAR_INT2: VersionedTVar<u64> =
                VersionedTVar::new(7);
        }
    }

    // A simple test where one thread alters the state of two tvars and another
    // thread reads that state.

    #[test]
    fn test1_simple_tvar_use() -> Result<(), TxnErrStatus> {
        use test1_state::*;
        let thread1 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                let tvar1_key = txn.capture_tvar(&TVAR_INT1)?;
                let tvar2_key = txn.capture_tvar(&TVAR_INT2)?;

                let mut tvar1_mut = tvar1_key.get_captured_tvar_mut();
                let mut tvar2_mut = tvar2_key.get_captured_tvar_mut();
                let sum = *tvar1_mut + *tvar2_mut;
                *tvar1_mut = sum;
                *tvar2_mut = 0;
                Ok(sum)
            })
        });
        assert_eq!(12, thread1.join().ok().unwrap());
        let thread2 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                let tvar1_key = txn.capture_tvar(&TVAR_INT1)?;
                let tvar2_key = txn.capture_tvar(&TVAR_INT2)?;

                let tvar1_ref = tvar1_key.get_captured_tvar_ref();
                assert_eq!(12, *tvar1_ref);

                let tvar2_ref = tvar2_key.get_captured_tvar_ref();
                assert_eq!(0, *tvar2_ref);

                Ok(())
            })
        });
        thread2.join().ok().unwrap();
        Ok(())
    }

    mod test2_state {
        use super::*;
        lazy_static! {
            pub static ref TVAR_INT1: VersionedTVar<u64> =
                VersionedTVar::new(0);
            pub static ref TVAR_INT2: VersionedTVar<u64> =
                VersionedTVar::new(0);

            // These mutexes are to ensure that the threads we create below
            // finish in the order we intend.
            pub static ref MUTEX1: Mutex<bool> = Mutex::new(false);
            pub static ref MUTEX2: Mutex<bool> = Mutex::new(false);
            pub static ref CONDVAR1: Condvar = Condvar::new();
            pub static ref CONDVAR2: Condvar = Condvar::new();
        }
    }

    #[test]
    fn test2_txn_write_conflict() {
        use test2_state::*;
        let thread1 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                let (tvar1_key, tvar2_key) = {
                    let mut lock1_guard = MUTEX1.lock().unwrap();
                    let key_pair =
                        (txn.capture_tvar(&TVAR_INT1)?,
                         txn.capture_tvar(&TVAR_INT2)?);
                    *lock1_guard = true;

                    // Notify, as we have captured the original state in
                    // thread1 and have set ourselves up for a conflict.
                    CONDVAR1.notify_all();
                    key_pair
                };

                {
                    let mut lock2_guard = MUTEX2.lock().unwrap();
                    // Wait until the second thread has completed its
                    // transaction
                    while !*lock2_guard {
                        lock2_guard =
                            CONDVAR2.wait(lock2_guard).unwrap();
                    }
                }

                *tvar1_key.get_captured_tvar_mut() += 3;
                *tvar2_key.get_captured_tvar_mut() += 7;

                Ok(())
            });

        });
        let thread2 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                {
                    let mut lock1_guard = MUTEX1.lock().unwrap();
                    while !*lock1_guard {
                        lock1_guard = CONDVAR1.wait(lock1_guard).unwrap();
                    }

                    *txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_mut() = 2;
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_mut() = 5;
                }
                Ok(())
            });
            // Now that the transaction on thread2 has completed, notify
            // thread1.
            let mut lock2_guard = MUTEX2.lock().unwrap();
            *lock2_guard = true;
            CONDVAR2.notify_all();
        });

        thread1.join().unwrap();
        thread2.join().unwrap();

        // Now, after the threads above, check to make sure that the contents of
        // the tvars are as we expect.
        let (tvar1_result, tvar2_result) =
            VersionedTransaction::start_txn(|txn| {
                Ok((*txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_ref(),
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_ref()))
            });
        assert_eq!(5, tvar1_result);
        assert_eq!(12, tvar2_result);
    }

    mod test3_state {
        use super::*;
        lazy_static! {
            pub static ref TVAR_INT1: VersionedTVar<u8> =
                VersionedTVar::new(17);
            pub static ref TVAR_INT2: VersionedTVar<u8> =
                VersionedTVar::new(25);

            // These mutexes are to ensure that the threads we create below
            // finish in the order we intend.
            pub static ref MUTEX1: Mutex<bool> = Mutex::new(false);
            pub static ref MUTEX2: Mutex<bool> = Mutex::new(false);
            pub static ref CONDVAR1: Condvar = Condvar::new();
            pub static ref CONDVAR2: Condvar = Condvar::new();
        }
    }

    // This test tests a very important aspect of this scheme: a transaction
    // that only reads tvars is not forced to restart by transactions that write
    // to their captured tvars after their tvars have been captured but before
    // they have finished. It is correct to allow them to finish, as the read
    // transaction can be said to have returned an accurate reflection of the
    // state before the write transaction occured and there are no guarantees
    // about what order these transactions will occur in anyway.
    #[test]
    fn test3_read_txn_does_not_conflict() {
        use test3_state::*;
        let thread1 = thread::spawn(|| {
            let (tvar1_result, tvar2_result) =
                VersionedTransaction::start_txn(|txn| {
                    let (tvar1_key, tvar2_key) = {
                        let mut lock1_guard =
                            MUTEX1.lock().unwrap();
                        let key_pair =
                            (txn.capture_tvar(&TVAR_INT1)?,
                            txn.capture_tvar(&TVAR_INT2)?);
                        *lock1_guard = true;

                        // Notify, as we have captured the original state in
                        // thread1 and have set ourselves up for a conflict.
                        CONDVAR1.notify_all();
                        key_pair
                    };

                    {
                        let mut lock2_guard =
                            MUTEX2.lock().unwrap();
                        // Wait until the second thread has completed its
                        // transaction
                        while !*lock2_guard {
                            lock2_guard =
                                CONDVAR2.wait(lock2_guard).unwrap();
                        }
                    }

                    Ok((*tvar1_key.get_captured_tvar_ref(),
                        *tvar2_key.get_captured_tvar_ref()))
                });
            assert_eq!(17, tvar1_result);
            assert_eq!(25, tvar2_result);
        });
        let thread2 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                {
                    let mut lock1_guard = MUTEX1.lock().unwrap();
                    while !*lock1_guard {
                        lock1_guard = CONDVAR1.wait(lock1_guard).unwrap();
                    }

                    *txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_mut() += 1;
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_mut() += 1;
                }
                Ok(())
            });
            // Now that the transaction on thread2 has completed, notify
            // thread1.
            let mut lock2_guard = MUTEX2.lock().unwrap();
            *lock2_guard = true;
            CONDVAR2.notify_all();
        });

        thread1.join().unwrap();
        thread2.join().unwrap();

        // Now, after the threads above, check to make sure that the contents of
        // the tvars are as we expect.
        let (tvar1_result, tvar2_result) =
            VersionedTransaction::start_txn(|txn| {
                Ok((*txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_ref(),
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_ref()))
            });
        assert_eq!(18, tvar1_result);
        assert_eq!(26, tvar2_result);

    }

    // This test is a bit more serious. Unlike the other transactions above,
    // here we're going to start a number of threads and have them attempt to
    // add their contents to a linked list in a sorted order. Note that because
    // this is a linked list we're talking about, the change that they make will
    // be considered dependent upon all prior nodes in the list plus the one
    // that they insert directly before.

    #[derive(Copy, Clone)]
    pub struct TVarLinkedListNode<PayloadType: Copy + 'static> {
        next: TVarCell<Option<TVarLinkedListNode<PayloadType>>>,
        payload: PayloadType
    }

    mod test4_state {
        use super::*;

        lazy_static! {
            pub static ref LIST_HEAD:
                VersionedTVar<Option<TVarLinkedListNode<u64>>> =
                    VersionedTVar::new(None);
        }
    }

    fn add_number_to_list_inner
        (txn: &VersionedTransaction,
         current_node_ref: &TVarLinkedListNode<u64>,
         num: u64)
            -> Result<(), TxnErrStatus>
    {
        assert!(current_node_ref.payload < num);
        let captured_next =
            txn.capture_tvar_cell(&current_node_ref.next)?;

        let replace_next = {
            let next_ref = captured_next.get_captured_tvar_ref();
            match *next_ref {
                None => true,
                Some(node) => {
                    node.payload > num
                }
            }
        };

        if replace_next {
            let mut next_mut = captured_next.get_captured_tvar_mut();
            *next_mut =
                Some(TVarLinkedListNode {
                    payload: num,
                    next: TVarCell::new(*next_mut)
                });
            return Ok(());
        }
        let next_captured_ref =
            captured_next.get_captured_tvar_ref();

        let next_ref: &Option<TVarLinkedListNode<u64>> =
            next_captured_ref.borrow();
        add_number_to_list_inner
            (txn,
             &next_ref.unwrap(),
             num)
    }

    fn add_number_to_list(num: u64) {
        use test4_state::*;
        VersionedTransaction::start_txn(|txn| {
            let current_node_capture = txn.capture_tvar(&LIST_HEAD)?;

            // First, check if the list is empty. If it is, we're going to
            // update it, so we might as well grab it as mut. Return whether or
            // not we updated the head.
            let insert_at_head = {
                let list_head_ref =
                    current_node_capture.get_captured_tvar_ref();
                match *list_head_ref {
                    Some(present_head_node) =>
                        num < present_head_node.payload,
                    None => true
                }
            };
            if insert_at_head {
                let mut list_head_mut =
                    current_node_capture.get_captured_tvar_mut();
                *list_head_mut =
                    Some(TVarLinkedListNode {
                             payload: num,
                             next: TVarCell::new(*list_head_mut)
                        });
                return Ok(());
            }

            // Once we have decided not to insert at the head, we can enter a
            // more sane common case: point at a linked list node and decide
            // whether to insert after it.
            let head_captured_ref =
                current_node_capture.get_captured_tvar_ref();

            let head_ref: &Option<TVarLinkedListNode<u64>> =
                head_captured_ref.borrow();

            add_number_to_list_inner(txn, &head_ref.unwrap(), num)
        })
    }

    fn start_thread_to_insert(num: u64) -> JoinHandle<()> {
        thread::spawn(move || {
            add_number_to_list(num)
        })
    }

    fn verify_list_inner
        (txn: &VersionedTransaction,
         curr_node: &TVarLinkedListNode<u64>,
         expected_num: u64)
    {
        print!("{}", curr_node.payload);
        assert_eq!(expected_num, curr_node.payload);
        let next_val_cache_key =
            txn.capture_tvar_cell(&curr_node.next).unwrap();
        let next_val_captured_ref = next_val_cache_key.get_captured_tvar_ref();
        let next_val_ref = next_val_captured_ref.borrow();
        match *next_val_ref {
            None => {
                println!(" -> _");
                assert_eq!(9, expected_num);
            },
            Some(linked_list_node) => {
                print!(" -> ");
                verify_list_inner(txn, &linked_list_node, expected_num + 1);
            }
        }
    }

    fn verify_linked_list_result() {
        use test4_state::*;
        VersionedTransaction::start_txn(|txn|{
            let list_head_ref =
                txn.capture_tvar(&LIST_HEAD)?.get_captured_tvar_ref();
            let head_node = list_head_ref.unwrap();
            verify_list_inner(txn, &head_node, 0);
            Ok(())
        })
    }

    #[test]
    fn test4_multi_thread_insert_sorted_linked_list() {
        // Take a list of items and, for each one, start a thread that opens a
        // transaction to insert it into the linked list.
        let vec_of_threads: Vec<JoinHandle<()>> =
            vec![7, 3, 8, 9, 0, 6, 2, 1, 4, 5]
                .into_iter()
                .map(|num|{start_thread_to_insert(num)})
                .collect();
        for handle in vec_of_threads {
            handle.join().unwrap();
        }

        // At this point, all of the insertion threads should have finished.
        // Now it should all be sorted.
        verify_linked_list_result();
    }

    mod test5_state {
        use super::*;
        lazy_static! {
            pub static ref TVAR_INT1: VersionedTVar<u64> =
                VersionedTVar::new(3);
            pub static ref TVAR_INT2: VersionedTVar<u64> =
                VersionedTVar::new(4);

            // These mutexes are to ensure that the threads we create below
            // finish in the order we intend.
            pub static ref MUTEX1: Mutex<bool> = Mutex::new(false);
            pub static ref MUTEX2: Mutex<bool> = Mutex::new(false);
            pub static ref CONDVAR1: Condvar = Condvar::new();
            pub static ref CONDVAR2: Condvar = Condvar::new();
        }
    }

    #[test]
    fn test5_txn_stale_read() {
        use test5_state::*;
        let thread1 = thread::spawn(|| {
            let txn_result = VersionedTransaction::start_txn(|txn| {
                let tvar1_key = {
                    let mut lock1_guard = MUTEX1.lock().unwrap();
                    let tvar1_key  = txn.capture_tvar(&TVAR_INT1)?;
                    *lock1_guard = true;

                    // Notify, as we have captured the original state in
                    // thread1 and have set ourselves up for a conflict.
                    CONDVAR1.notify_all();
                    tvar1_key
                };

                {
                    let mut lock2_guard = MUTEX2.lock().unwrap();
                    // Wait until the second thread has completed its
                    // transaction
                    while !*lock2_guard {
                        lock2_guard =
                            CONDVAR2.wait(lock2_guard).unwrap();
                    }
                }
                let tvar2_key = txn.capture_tvar(&TVAR_INT2)?;

                Ok(*tvar1_key.get_captured_tvar_ref() +
                   *tvar2_key.get_captured_tvar_ref())
            });

            assert_eq!(txn_result, 9);
        });
        let thread2 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                {
                    let mut lock1_guard = MUTEX1.lock().unwrap();
                    while !*lock1_guard {
                        lock1_guard = CONDVAR1.wait(lock1_guard).unwrap();
                    }

                    *txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_mut() += 1;
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_mut() += 1;
                }
                Ok(())
            });
            // Now that the transaction on thread2 has completed, notify
            // thread1.
            let mut lock2_guard = MUTEX2.lock().unwrap();
            *lock2_guard = true;
            CONDVAR2.notify_all();
        });

        thread1.join().unwrap();
        thread2.join().unwrap();

        // Now, after the threads above, check to make sure that the contents of
        // the tvars are as we expect.
        let (tvar1_result, tvar2_result) =
            VersionedTransaction::start_txn(|txn| {
                Ok((*txn.capture_tvar(&TVAR_INT1)?.get_captured_tvar_ref(),
                    *txn.capture_tvar(&TVAR_INT2)?.get_captured_tvar_ref()))
            });
        assert_eq!(4, tvar1_result);
        assert_eq!(5, tvar2_result);
    }

}
