use indexmap::IndexMap;
use indexmap::map::Entry::Occupied;
use indexmap::map::Entry::Vacant;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;
use std::sync::Mutex;
use std::ptr::NonNull;
use std::any::TypeId;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::spin_loop_hint;
use std::marker::PhantomData;
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::null_mut;
use lazy_static::lazy_static;
use std::thread_local;
use std::cell::RefCell;
use std::cell::Cell;
use std::result::Result;
use std::u64::MAX;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ops::DerefMut;
use crossbeam_epoch::Owned;
use crossbeam_epoch::Shared;
use crossbeam_epoch::Guard;

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

struct SendSyncPointerWrapper<PointeeType> {
    ptr: * mut PointeeType
}

impl<PointeeType> Clone for SendSyncPointerWrapper<PointeeType> {
    fn clone(&self) -> Self {
        SendSyncPointerWrapper {
            ptr: self.ptr
        }
    }
}

impl<PointeeType> Copy for SendSyncPointerWrapper<PointeeType> { }

unsafe impl<PointeeType> Send for SendSyncPointerWrapper<PointeeType> { }
unsafe impl<PointeeType> Sync for SendSyncPointerWrapper<PointeeType> { }

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
    timestamp: AtomicU64,
    next_ptr: * mut TVarVersionInner<()>,
    // We will often be referring to these inner versions as
    // TVarVersionInner<()> due to losing track of types. This is the type id of
    // the TVarVersionInner<GuardedType>, allowing us to check our work to
    // prevent confusion and to automatically look up functions associated with
    // this guarded type (for instance, the destructor).
    with_guarded_type_id: TypeId,
    dtor_fn: fn(DynamicInnerTVarPtr),
    dealloc_fn: fn(DynamicInnerTVarPtr),
    inner_struct: GuardedType
}

fn clean_up_tvar_version_inner<GuardedType: 'static>
    (untyped_version: DynamicInnerTVarPtr)
{
    let opt_typed_version =
        untyped_version.dynamic_cast_guarded::<GuardedType>();
    match opt_typed_version {
        Some(typed_version) => {
            let mut mut_typed_version = typed_version;
            unsafe {
                let payload_ref = &mut mut_typed_version.as_mut().inner_struct;
                std::ptr::drop_in_place(payload_ref)
            };
        },
        None => {
            panic!
                ("Type confusion while trying to clean up a TVarVersionInner!");
        }
    }
}

fn dealloc_tvar_version_inner<GuardedType: 'static>
    (untyped_version: DynamicInnerTVarPtr)
{
    let opt_typed_version =
        untyped_version.dynamic_cast_guarded::<GuardedType>();
    match opt_typed_version {
        Some(typed_version) => {
            let guard = crossbeam_epoch::pin();
            unsafe {
                guard.defer_destroy
                    (Shared::from
                        (typed_version.as_ptr()
                            as * const TVarVersionInner<GuardedType>));
            }
        },
        None => {
            panic!
                ("Type confusion while trying to clean up a TVarVersionInner!");
        }
    }
}

impl<GuardedType: 'static> TVarVersionInner<GuardedType> {

    fn new(new_val: GuardedType) -> TVarVersionInner<GuardedType> {
        TVarVersionInner {
            timestamp: AtomicU64::new(TXN_COUNTER_NON_CANON),
            next_ptr: null_mut(),
            with_guarded_type_id:
                TypeId::of::<TVarVersionInner<GuardedType>>(),
            dtor_fn: clean_up_tvar_version_inner::<GuardedType>,
            dealloc_fn: dealloc_tvar_version_inner::<GuardedType>,
            inner_struct: new_val
        }
    }

    fn alloc(new_val: GuardedType) -> NonNull<TVarVersionInner<GuardedType>> {
        idempotent_add_version_guarded_type::<GuardedType>();
        let owned_new_version = Owned::new(TVarVersionInner::new(new_val));
        let guard = crossbeam_epoch::pin();
        let shared_new_version = owned_new_version.into_shared(&guard);
        NonNull::new
            (shared_new_version.as_raw() as * mut TVarVersionInner<GuardedType>)
            .unwrap()
            .cast::<TVarVersionInner<GuardedType>>()
    }
}

lazy_static! {
    static ref ID_TO_LAYOUT_MAP:
        RwLock<HashMap<TypeId, Layout>> = RwLock::new(HashMap::new());
}

fn idempotent_add_version_guarded_type<T: 'static>() {
    let this_type_typeid = TypeId::of::<T>();
    let this_type_layout = Layout::new::<T>();
    {
        let read_lock_guard = ID_TO_LAYOUT_MAP.read().unwrap();
        let map_lookup_result = read_lock_guard.get(&this_type_typeid);
        match map_lookup_result {
            Some(layout) => {
                assert_eq!(*layout, this_type_layout);
                return;
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
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct DynamicInnerTVarPtr(NonNull<TVarVersionInner<()>>);

impl DynamicInnerTVarPtr {

    fn dynamic_cast_guarded<GuardedType: 'static>(self)
        -> Option<NonNull<TVarVersionInner<GuardedType>>> {
        let target_type_id = TypeId::of::<TVarVersionInner<GuardedType>>();
        if target_type_id == unsafe { self.0.as_ref().with_guarded_type_id } {
            Some(self.0.cast::<TVarVersionInner<GuardedType>>())
        } else {
            None
        }
    }

    fn get_guarded_ref<GuardedType: 'static>(&self) -> &'static GuardedType {
        &(
            unsafe {
                &*self.dynamic_cast_guarded::<GuardedType>().unwrap().as_ptr()
            }
        ).inner_struct
    }

    fn get_guarded_mut<GuardedType: 'static>(&mut self)
        -> &'static mut GuardedType
    {
        &mut (
            unsafe {
                &mut *self
                    .dynamic_cast_guarded::<GuardedType>().unwrap().as_ptr()
            }
        ).inner_struct
    }

    fn wrap_static_inner_version<GuardedType: 'static>
        (nonnull_inner_version: NonNull<TVarVersionInner<GuardedType>>)
        -> DynamicInnerTVarPtr
    {
        DynamicInnerTVarPtr {
            0: nonnull_inner_version.cast::<TVarVersionInner<()>>()
        }
    }

    fn has_guarded_type<GuardedType: 'static>(&self) -> bool {
        self.dynamic_cast_guarded::<GuardedType>().is_some()
    }

    // Gets the timestamp on a pointee. This version just naively returns the
    // value that is present.
    fn get_pointee_timestamp_val(self) -> u64 {
        unsafe {
            // This must be an acquire load. We may be loading a version that
            // does not happen-before this transaction, in which case we must
            // acquire to be able to inspect the timestamp that it points to.
            self.0.as_ref().timestamp.load(Acquire)
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

unsafe impl Send for DynamicInnerTVarPtr { }
unsafe impl Sync for DynamicInnerTVarPtr { }

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

// An allocator for caching TVar versions in such a fashion that they can be
// re-allocated quickly without synchronization.
struct TVarVersionAllocator {
    layout: Layout,
    // This is the list of objects that are free and ready to serve the layout
    // indicated.
    free_list: AtomicPtr<TVarVersionInner<()>>,
}

impl TVarVersionAllocator {
    fn new(layout: Layout) -> TVarVersionAllocator {
        TVarVersionAllocator {
            layout,
            free_list: AtomicPtr::new(null_mut())
        }
    }

    fn alloc_shadow_version<GuardedType: 'static>(&self, init: GuardedType)
        -> TVarShadowVersion<GuardedType>
    {
        assert!(self.layout.size() >=
            std::mem::size_of::<TVarVersionInner<GuardedType>>());
        assert!(self.layout.align() >=
            std::mem::align_of::<TVarVersionInner<GuardedType>>());

        let mut free_list_head_ptr = self.free_list.load(Acquire);
        let tvar_ptr = loop {
            // If the head is equal to null, allocate a new item.
            if free_list_head_ptr == null_mut() {
                break TVarVersionInner::alloc(init)
                    .cast::<TVarVersionInner<()>>();
            }
            let free_list_head =
                unsafe { free_list_head_ptr.as_mut() }.unwrap();
            // Otherwise, try to grab the current head of the free list.
            let head_next = free_list_head.next_ptr;
            let cmp_ex_result =
                self.free_list.compare_exchange_weak
                    (free_list_head, head_next, Acquire, Acquire);
            match cmp_ex_result {
                Ok(reserved_ptr) => {
                    let uninit_ptr =
                        reserved_ptr
                            as * mut MaybeUninit<TVarVersionInner<GuardedType>>;
                    let uninit_mut = unsafe { uninit_ptr.as_mut() }.unwrap();
                    *uninit_mut =
                        MaybeUninit::new(TVarVersionInner::new(init));
                    break NonNull::new
                        (uninit_ptr as * mut TVarVersionInner<()>).unwrap();
                },
                Err(seen_head) => {
                    free_list_head_ptr = seen_head;
                }
            }
        };
        TVarShadowVersion {
            type_erased_inner: DynamicInnerTVarPtr {
                0: tvar_ptr
            },
            phantom: PhantomData
        }
    }

    fn return_stale_version_pointer(&self, stale_ptr: DynamicInnerTVarPtr) {
        let dtor_fn = unsafe { stale_ptr.0.as_ref().dtor_fn };
        dtor_fn(stale_ptr);

        let mut current_free_list_head = self.free_list.load(Relaxed);
        let mut mut_stale_ptr = stale_ptr;
        loop {
            unsafe {
                mut_stale_ptr.0.as_mut().next_ptr = current_free_list_head;
            }
            let cmp_ex_result =
                self.free_list.compare_exchange_weak
                    (current_free_list_head,
                     stale_ptr.0.as_ptr(),
                     Release,
                     Relaxed);
            match cmp_ex_result {
                Ok(_) => { break; }
                Err(seen_head) => { current_free_list_head = seen_head; }
            }
        }
    }

    fn return_formerly_canon_pointer
        (&'static self, returned: DynamicInnerTVarPtr)
    {
        // For a formerly canon version, we have to wait until all threads
        // currently running transactions have finished possibly using this
        // version before it can be reused.
        let guard = crossbeam_epoch::pin();
        guard.defer(move || {
            self.return_stale_version_pointer(returned);
        })
    }
}

#[derive(PartialEq, Eq, Copy, Clone)]
struct HashableLayout(Layout);

impl std::hash::Hash for HashableLayout {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_usize(self.0.align());
        hasher.write_usize(self.0.size());
    }
}

// This global allocator map stores a pointer to the allocator to be used for
// each layout. These allocators are allocated upon the first request of a
// particular layout and will not be replaced. As such, they will be cached in
// each thread's transaction state and this will only be consulted the first
// time a thread needs to allocate an object with a particular layout. Thus, I
// don't think using a lock here dilutes the lock-free claims of this algorithm.
lazy_static! {
    static ref GLOBAL_LAYOUT_TO_ALLOC_MAP:
        Mutex<
            HashMap<HashableLayout,
                    SendSyncPointerWrapper<TVarVersionAllocator>>> =
            Mutex::new(HashMap::new());
}

thread_local! {
    static THREAD_LAYOUT_TO_ALLOC_MAP:
        RefCell<HashMap<HashableLayout, NonNull<TVarVersionAllocator>>> =
            RefCell::new(HashMap::new());
}

fn get_or_create_version_allocator_for_layout(layout: Layout)
    -> NonNull<TVarVersionAllocator>
{
    use std::collections::hash_map::Entry;

    // Make sure everything is aligned to 8 bytes.
    assert!(layout.align() >= 8);

    let hashable_layout = HashableLayout { 0: layout };

    let thread_local_lookup_result =
        THREAD_LAYOUT_TO_ALLOC_MAP.with(|alloc_map_key| {
            alloc_map_key
                .borrow()
                .get(&hashable_layout)
                .map(|found_val_ref|{ *found_val_ref })
        });

    match thread_local_lookup_result {
        Some(allocator_ptr) => return allocator_ptr,
        None => {
            // We have to look this up in the global map.
        }
    }

    // We should only get here the first time a thread tries to get a particular
    // layout. The thread-local map should handle it otherwise.
    let mut map_mutex_guard = GLOBAL_LAYOUT_TO_ALLOC_MAP.lock().unwrap();
    let layout_entry = map_mutex_guard.entry(hashable_layout);
    let result_ptr = match layout_entry {
        Entry::Occupied(present_entry) => {
            NonNull::new(present_entry.get().ptr).unwrap()
        },
        Entry::Vacant(vacant_entry) => {
            let entry_ptr =
                Box::leak(Box::new(TVarVersionAllocator::new(layout)));
            vacant_entry.insert(SendSyncPointerWrapper { ptr: entry_ptr });
            NonNull::new(entry_ptr).unwrap()
        }
    };

    THREAD_LAYOUT_TO_ALLOC_MAP.with(|alloc_map_key| {
        alloc_map_key.borrow_mut().insert(hashable_layout, result_ptr);
    });
    result_ptr
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

    fn get_dyn_inner_version_ptr(&self) -> DynamicInnerTVarPtr {
        self.type_erased_inner
    }

}

#[derive(Copy, Clone)]
struct VersionedTVarTypeErasedRef {
    tvar_ptr: SendSyncPointerWrapper<VersionedTVarTypeErased>
}

struct VersionedTVarRef<GuardedType> {
    tvar_ref: VersionedTVarTypeErasedRef,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType> Clone for VersionedTVarRef<GuardedType> {
    fn clone(&self) -> Self {
        VersionedTVarRef {
            tvar_ref: self.tvar_ref,
            phantom: PhantomData
        }
    }
}

impl<GuardedType> Copy for VersionedTVarRef<GuardedType> { }

// This is a ref which owns the tvar version contained within. Once all such
// refs are dropped, the tvar is freed.
pub struct SharedTVarRef<GuardedType> {
    tvar_ref: VersionedTVarRef<GuardedType>
}

impl<GuardedType> Drop for SharedTVarRef<GuardedType> {
    fn drop(&mut self) {
        let tvar_ptr = self.tvar_ref.tvar_ref.tvar_ptr.ptr;
        if tvar_ptr == null_mut() {
            return;
        }
        let tvar = unsafe { &mut *tvar_ptr };
        let sub_prior = tvar.refct.fetch_sub(1, Relaxed);
        assert_ne!(sub_prior, NON_REFCT_TVAR);
        let new_refct = sub_prior - 1;
        if new_refct == 0 {
            let guard = crossbeam_epoch::pin();
            tvar.retire();
            let const_ptr =
                self.tvar_ref.tvar_ref.tvar_ptr.ptr as
                    * const VersionedTVarTypeErased;
            let crossbeam_shared_ptr = Shared::from(const_ptr);
            unsafe {
                guard.defer_destroy(crossbeam_shared_ptr);
            }
        }
    }
}

impl<GuardedType> Clone for SharedTVarRef<GuardedType> {
    fn clone(&self) -> Self {
        let tvar_ptr = self.tvar_ref.tvar_ref.tvar_ptr.ptr;
        if tvar_ptr != null_mut() {
            let tvar = unsafe { &mut *tvar_ptr };
            let add_prior = tvar.refct.fetch_add(1, Relaxed);
            assert_ne!(add_prior, NON_REFCT_TVAR);
            assert_ne!(add_prior, 0);
        }

        SharedTVarRef {
            tvar_ref: self.tvar_ref
        }
    }
}

struct CanonPtrAndWriteReserved {
    canon_ptr: DynamicInnerTVarPtr,
    write_reserved: bool
}

impl CanonPtrAndWriteReserved {
    fn new(dyn_pointer: DynamicInnerTVarPtr, write_reserved: bool)
        -> CanonPtrAndWriteReserved
    {
        let canon_ptr_as_int = dyn_pointer.0.as_ptr() as u64;
        // Assert that the bottom bit of the pointer is 0. If this isn't the
        // case, we can't pack it.
        assert!(canon_ptr_as_int & 1 == 0);

        CanonPtrAndWriteReserved {
            canon_ptr: dyn_pointer,
            write_reserved
        }
    }

    fn pack(&self) -> u64 {
        let canon_ptr_as_int = self.canon_ptr.0.as_ptr() as u64;
        // Assert that the bottom bit of the pointer is 0.
        assert!(canon_ptr_as_int & 1 == 0);
        canon_ptr_as_int | if self.write_reserved { 1 } else { 0 }
    }

    fn unpack(packed: u64) -> CanonPtrAndWriteReserved {
        let write_reserved = (packed & 1) == 1;
        let canon_raw = ((packed >> 1) << 1) as * mut TVarVersionInner<()>;
        let canon_nonnull = NonNull::new(canon_raw).unwrap();
        let canon_ptr =
            DynamicInnerTVarPtr::wrap_static_inner_version(canon_nonnull);
        CanonPtrAndWriteReserved::new(canon_ptr, write_reserved)
    }
}

const NON_REFCT_TVAR: u64 = MAX;

const RETIRED_CANON_OBJECT: u8 = 0;
const RETIRED_CANON_PTR: * const u8 = &RETIRED_CANON_OBJECT as * const u8;

struct VersionedTVarTypeErased {
    packed_canon_ptr: AtomicU64,
    refct: AtomicU64,
    allocator: &'static TVarVersionAllocator
}

impl VersionedTVarTypeErased {
    fn fetch_and_unpack_canon_ptr(&self, ordering: Ordering)
            -> CanonPtrAndWriteReserved {
        let packed_canon_ptr = self.packed_canon_ptr.load(ordering);
        CanonPtrAndWriteReserved::unpack(packed_canon_ptr)
    }

    fn has_canon_version_ptr(&self, candidate: DynamicInnerTVarPtr) -> bool {
        self.fetch_and_unpack_canon_ptr(Relaxed).canon_ptr == candidate
    }

    fn get_current_canon_version(&self) -> DynamicInnerTVarPtr {
        let canon_ptr_and_write_reserved =
            self.fetch_and_unpack_canon_ptr(Acquire);

        canon_ptr_and_write_reserved.canon_ptr
    }

    fn clear_write_reservation(&self) {
        let all_bits_except_bottom: u64 = !1;
        self.packed_canon_ptr.fetch_and(all_bits_except_bottom, Relaxed);
    }

    // This is used to prepare a TVar to be deallocated and reused. This
    // function assumes that the TVar is no longer reachable (or at least that
    // all threads know that this TVar should no longer be accessed), and thus
    // the only way to access the contents is through transactions that started
    // before retire was called. It swaps in a RETIRED marker value as the canon
    // version and prepares the former canon version to be placed on the free
    // list. Any transaction that sees this RETIRED marker will restart its
    // transaction and, if hitting the RETIRED marker a second time, will panic.
    fn retire(&self) {
        let former_canon_ptr = {
            let mut current_packed_canon_ptr =
                self.packed_canon_ptr.load(Relaxed);
            loop {
                let current_canon_ptr =
                    CanonPtrAndWriteReserved::unpack(current_packed_canon_ptr);
                // We should not replace a write-reserved canon_ptr, as the
                // transaction infrastructure expects that the canon value is,
                // well, reserved. If we were to swap in the RETIRED value, the
                // currently-running transaction would swap right in over the
                // top of it. Instead, try to compare and exchange with the
                // unreserved equivalent. If the writing transaction had to
                // back off, this may succeed and we can retire the tvar. If
                // not, we'll get the canon pointer in the failure value and
                // can just try again.
                let current_canon_unreserved = CanonPtrAndWriteReserved {
                    canon_ptr: current_canon_ptr.canon_ptr,
                    write_reserved: false
                };
                let cmp_ex_result =
                    self.packed_canon_ptr.compare_exchange_weak
                        (current_canon_unreserved.pack(),
                        RETIRED_CANON_PTR as u64,
                        Relaxed,
                        Relaxed);
                match cmp_ex_result {
                    Ok(_) => { break current_canon_unreserved.canon_ptr; }
                    Err(found_packed_ptr) => {
                        current_packed_canon_ptr = found_packed_ptr;
                        spin_loop_hint();
                        continue;
                    }
                }
            }
        };

        self.allocator.return_formerly_canon_pointer(former_canon_ptr);
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

        let allocator_ptr =
            get_or_create_version_allocator_for_layout
                (Layout::new::<TVarVersionInner<GuardedType>>());

        let allocator_ref = unsafe { &*allocator_ptr.as_ptr() };
        let init_shadow_version = allocator_ref.alloc_shadow_version(inner_val);

        let init_type_erased_inner = init_shadow_version.type_erased_inner;

        let mut inner_version_ptr =
            init_type_erased_inner
                .dynamic_cast_guarded::<GuardedType>()
                .unwrap();

        unsafe {
            inner_version_ptr
                .as_mut().timestamp.store(TXN_COUNTER_INIT, Relaxed);
        }

        let init_dynamic_inner_version_ptr = init_type_erased_inner;
        let canon_ptr_and_write_reserved =
            CanonPtrAndWriteReserved::new
                (init_dynamic_inner_version_ptr, false);
        let result: VersionedTVar<GuardedType> =
            VersionedTVar {
                inner_type_erased: VersionedTVarTypeErased {
                    allocator: allocator_ref,
                    refct: AtomicU64::new(NON_REFCT_TVAR),
                    packed_canon_ptr:
                        AtomicU64::new(canon_ptr_and_write_reserved.pack())
                },
                phantom: PhantomData
            };
        result
    }

    pub fn new_shared(inner_val: GuardedType) -> SharedTVarRef<GuardedType> {
        let owned_ptr = Owned::new(VersionedTVar::new(inner_val));
        // Initially, the refct indicates that this is a non-refcounted tvar.
        // Update it to make it have a refcount of 1.
        (*owned_ptr).inner_type_erased.refct.store(1, Relaxed);
        let guard = crossbeam_epoch::pin();
        let shared_ptr = owned_ptr.into_shared(&guard);
        let raw_tvar_ptr = shared_ptr.as_raw();
        let raw_mut_tvar_ptr = raw_tvar_ptr as * mut VersionedTVar<GuardedType>;
        let type_erased_inner_ptr =
            unsafe {
                &mut ((*raw_mut_tvar_ptr).inner_type_erased)
                    as * mut VersionedTVarTypeErased
                };
        SharedTVarRef {
            tvar_ref: VersionedTVarRef {
                tvar_ref: VersionedTVarTypeErasedRef {
                    tvar_ptr:
                        SendSyncPointerWrapper { ptr: type_erased_inner_ptr }
                },
                phantom: PhantomData

            }
        }
    }

    pub fn retire(&self) {
        self.inner_type_erased.retire();
    }

}

struct CapturedTVarTypeErased {
    captured_version_ptr: DynamicInnerTVarPtr,
    shadow_copy_ptr: Option<DynamicInnerTVarPtr>,
    // The CapturedTVar acts like a RefCell, but not exactly, because we want to
    // free up the hash map for insertion while borrowing individual cells.
    mut_borrowed: Cell<bool>,
    num_shared_borrows: Cell<u64>,
    index: CapturedTVarIndex
}

impl CapturedTVarTypeErased {

    fn new<GuardedType: 'static>
        (captured_version: TVarImmVersion<GuardedType>,
         new_index: CapturedTVarIndex)
        -> CapturedTVarTypeErased
    {
        let dyn_inner_version_ptr =
            captured_version.get_dyn_inner_version_ptr();
        CapturedTVarTypeErased {
            captured_version_ptr: dyn_inner_version_ptr,
            shadow_copy_ptr: None,
            mut_borrowed: Cell::new(false),
            num_shared_borrows: Cell::new(0),
            index: new_index
        }
    }

    fn get_captured_version<GuardedType: 'static>(&self)
         -> TVarImmVersion<GuardedType>
    {
        let result_dynamic_inner_version_ptr = self.captured_version_ptr;
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
                assert!(present_shadow_copy_ptr
                            .has_guarded_type::<GuardedType>());
                Some(TVarShadowVersion {
                    type_erased_inner: present_shadow_copy_ptr,
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
                let tvar_mut_ref = unsafe { self.index.0.as_mut() };
                let new_shadow_version =
                    tvar_mut_ref
                        .allocator
                        .alloc_shadow_version
                            (captured_version
                                .get_dyn_inner_version_ptr()
                                .get_guarded_ref::<GuardedType>()
                                .tvar_version_clone());
                self.shadow_copy_ptr =
                    Some(new_shadow_version.type_erased_inner);
                new_shadow_version
            }
        }
    }

    fn discard_shadow_state(&mut self) {
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
struct CapturedTVarIndex(NonNull<VersionedTVarTypeErased>);

impl CapturedTVarIndex {
    fn get_nonnull(&self) -> NonNull<VersionedTVarTypeErased> {
        self.0
    }
}

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
            let mut per_thread_state_ref =
                per_thread_txn_state_key.borrow_mut();

            let captured_tvar_mut =
                &mut per_thread_state_ref
                    .captured_tvar_cache[&self.index.get_nonnull()]
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
            let mut per_thread_state_ref =
                per_thread_txn_state_key.borrow_mut();

            let captured_tvar_mut =
                &mut per_thread_state_ref
                    .captured_tvar_cache[&self.index.get_nonnull()]
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
                &per_thread_state_ref
                    .captured_tvar_cache[&self.cache_index.get_nonnull()];
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
    txn_version_acquisition_threshold: u64
}

impl PerThreadTransactionState {
    fn reset_txn_state(&mut self, drop_shadow_versions: bool) {
        if drop_shadow_versions {
            for (tvar_ptr, captured_tvar_cell) in
                self.captured_tvar_cache.drain(..)
            {
                let mut mut_captured_tvar = captured_tvar_cell.borrow_mut();
                let previous_shadow_copy =
                    mut_captured_tvar.shadow_copy_ptr.take();
                let mut mut_tvar_ptr = tvar_ptr;
                match previous_shadow_copy {
                    Some(present_shadow_copy) => {
                        unsafe { mut_tvar_ptr.as_mut() }
                            .allocator
                            .return_stale_version_pointer(present_shadow_copy);
                    },
                    None => {
                        // Nothing to drop
                    }
                }
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
    KnownFreshIndex,
    MaybeNotFreshContext(TVarImmVersion<GuardedType>, u64)
}

// This struct actually contains little transaction data. It acts as a gateway
// to the actual transaction state, which is stored in thread local storage.
// Because the per-thread transaction state cannot be accessed unless one is in
// a transaction, the pattern of allowed accesses is actually more intuitive if
// a user acts as if the transaction state were stored within the
// VersionedTransaction.
pub struct VersionedTransaction<'guard> {
    txn_succeeded: bool,
    _epoch_guard: &'guard Guard
}

impl<'guard> Drop for VersionedTransaction<'guard> {
    fn drop(& mut self) {
        // When a transaction has finished, we should always make sure that the
        // per-thread state is cleared.
        self.with_per_thread_txn_state_mut(|per_thread_state_mut|{
            let should_drop_shadow_versions = !self.txn_succeeded;
            per_thread_state_mut.reset_txn_state(should_drop_shadow_versions);
        })
    }
}

impl<'guard> VersionedTransaction<'guard> {

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
                captured_tvar.borrow().captured_version_ptr;
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

        // Now we must update all of the tvars we have touched. For those
        // that we did not modify, we must clear the write reserved bit. For
        // those that we did, we must swap in our shadow version upgraded to
        // a new immutable version.
        for (tvar_ptr, captured_tvar) in captured_tvar_cache_mut.iter() {
            let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
            match captured_tvar.borrow().shadow_copy_ptr {
                Some(inner_shadow_copy_ptr) => {
                    let raw_shadow_copy_ptr = inner_shadow_copy_ptr.0.as_ptr();
                    // Initially, set the timestamp for all new items being
                    // swapped in to UPDATING.
                    unsafe {
                        (*raw_shadow_copy_ptr)
                            .timestamp.store(TXN_COUNTER_UPDATING_VAL, Relaxed);
                    }

                    // This is our first time publishing this shadow version
                    // to other threads, so we must use a release ordering.
                    tvar_ref
                        .packed_canon_ptr
                        .store
                            (CanonPtrAndWriteReserved::new
                                (inner_shadow_copy_ptr, false).pack(),
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

        // Now that we have completed the commit, we still have to
        // perform some cleanup actions. For all captured tvars where we
        // swapped in a new value, we must place the old canonical version
        // on the stale list.
        for (tvar_ptr, captured_tvar) in captured_tvar_cache_mut.drain(..) {
            let mut shadow_copy_ptr =
                match captured_tvar.borrow().shadow_copy_ptr {
                    Some(shadow_copy_ptr) => shadow_copy_ptr,
                    // There's nothing to do for captured tvars which did not
                    // have a shadow copy (ie, were not writes.)
                    None => { continue; }
                };
            // Mark the former shadow copy (now the new canon version) with the
            // current txn number timestamp.
            unsafe {
                shadow_copy_ptr
                    .0
                    .as_mut()
                    .timestamp
                    .store(this_txn_time_number, Release);
            }
            // For writes, place the old canon version onto the stale
            // version list.
            let old_canon = captured_tvar.borrow().captured_version_ptr;
            let tvar_ref = unsafe {&mut (*tvar_ptr.as_ptr()) };
            tvar_ref.allocator.return_formerly_canon_pointer(old_canon);
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
        loop {
            let guard = crossbeam_epoch::pin();
            // A top-level transaction will borrow from the per-thread
            // transaction state and assume that no other transaction has
            // checked it out.
            let mut txn =
                VersionedTransaction {
                    txn_succeeded: false,
                    _epoch_guard: &guard
                };

            txn.with_per_thread_txn_state_mut(|per_thread_state_mut| {
                per_thread_state_mut.reset_txn_state
                    (true/*should_drop_shadow_versions*/);
            });
            let txn_fn_result = txn_fn(&mut txn);
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
                txn.with_per_thread_txn_state_ref
                    (|per_thread_state_ref| {
                         *per_thread_state_ref.is_write_txn.borrow()
                    });
            if is_write_txn {
                let commit_succeeded =
                    PER_THREAD_TXN_STATE.with(|per_thread_state_key| {
                        txn.perform_write_txn_commit
                            (&mut *per_thread_state_key.borrow_mut())
                    });
                if !commit_succeeded {
                    continue;
                }
            } else {
                txn.txn_succeeded = true;
            }

            return user_result;
        }
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

    pub fn capture_tvar_cell<GuardedType: TVarVersionClone + 'static>
        (&self, shared_tvar_ref: &SharedTVarRef<GuardedType>)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        let inner_tvar_ref = &shared_tvar_ref.tvar_ref;
        self.capture_type_erased_tvar
            (unsafe {&*inner_tvar_ref.tvar_ref.tvar_ptr.ptr})
    }

    pub fn capture_tvar
        <GuardedType: TVarVersionClone + 'static>
        (&self, tvar: &VersionedTVar<GuardedType>)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        self.capture_type_erased_tvar(&tvar.inner_type_erased)
    }


    fn capture_type_erased_tvar
        <GuardedType: TVarVersionClone + 'static>
        (&self, tvar: &VersionedTVarTypeErased)
            -> Result<CapturedTVarCacheKey<GuardedType>, TxnErrStatus>
    {
        let cache_key = NonNull::from(tvar);
        let captured_tvar_index = CapturedTVarIndex { 0: cache_key };
        let success_result =
            Ok(CapturedTVarCacheKey {
                    cache_index: captured_tvar_index,
                    phantom: PhantomData,
                    phantom_lifetime: PhantomData
                });

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
                    Occupied(_) => {
                        TVarIndexResult::KnownFreshIndex
                    },
                    Vacant(vacant_entry) => {
                        let versioned_tvar_type_erased =
                            unsafe { cache_key.as_ref() };
                        let current_canon =
                            versioned_tvar_type_erased
                                .get_current_canon_version();
                        let current_canon_tvar_imm_version = TVarImmVersion {
                            type_erased_inner: current_canon,
                            phantom: PhantomData
                        };
                        let captured_tvar =
                            CapturedTVarTypeErased::new
                                (current_canon_tvar_imm_version,
                                 captured_tvar_index);
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
                            TVarIndexResult::KnownFreshIndex
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

        match index_lookup_result {
            TVarIndexResult::KnownFreshIndex => {
                success_result
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
                    let previous_value_expect_none =
                        per_thread_txn_state_mut
                            .captured_tvar_cache
                            .insert
                                (cache_key,
                                    RefCell::new
                                    (CapturedTVarTypeErased::new
                                        (current_canon,
                                            captured_tvar_index)));
                    assert!(previous_value_expect_none.is_none());
                    success_result
                })
            }
        }
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
        TVAR_INT1.retire();
        TVAR_INT2.retire();
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

        TVAR_INT1.retire();
        TVAR_INT2.retire();
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

        TVAR_INT1.retire();
        TVAR_INT2.retire();
    }

    // This test is a bit more serious. Unlike the other transactions above,
    // here we're going to start a number of threads and have them attempt to
    // add their contents to a linked list in a sorted order. Note that because
    // this is a linked list we're talking about, the change that they make will
    // be considered dependent upon all prior nodes in the list plus the one
    // that they insert directly before.

    #[derive(Clone)]
    pub struct TVarLinkedListNode<PayloadType: Copy + 'static> {
        next: SharedTVarRef<Option<TVarLinkedListNode<PayloadType>>>,
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
            match &*next_ref {
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
                    next: VersionedTVar::new_shared(next_mut.clone())
                });
            return Ok(());
        }
        let next_captured_ref =
            captured_next.get_captured_tvar_ref();

        let next_ref: &Option<TVarLinkedListNode<u64>> =
            next_captured_ref.borrow();
        let next_node = next_ref.as_ref().unwrap();
        add_number_to_list_inner(txn, next_node, num)
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
                match &*list_head_ref {
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
                             next: VersionedTVar::new_shared
                                (list_head_mut.clone())
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

            add_number_to_list_inner(txn, &head_ref.as_ref().unwrap(), num)
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
        match next_val_ref {
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
            let head_node = list_head_ref.as_ref().unwrap();
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

        test4_state::LIST_HEAD.retire();
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

        TVAR_INT1.retire();
        TVAR_INT2.retire();
    }
}
