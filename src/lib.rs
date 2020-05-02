#![feature(alloc_layout_extra)]
#![feature(maybe_uninit_extra)]
#![feature(const_fn)]
#![feature(const_if_match)]
#![feature(const_raw_ptr_to_usize_cast)]
#![feature(const_panic)]

use indexmap::IndexMap;
use indexmap::map::Entry::Occupied;
use indexmap::map::Entry::Vacant;
use std::convert::AsRef;
use std::convert::AsMut;
use std::slice::from_raw_parts;
use std::slice::from_raw_parts_mut;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
use std::ptr::NonNull;
use std::any::TypeId;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::spin_loop_hint;
use std::marker::PhantomData;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::collections::HashMap;
use std::ptr::null_mut;
use std::ptr::drop_in_place;
use lazy_static::lazy_static;
use std::thread_local;
use std::cell::RefCell;
use std::result::Result;
use std::u64::MAX;
use std::mem::MaybeUninit;
use std::mem::size_of;
use std::mem::align_of;
use std::mem::transmute;
use std::ops::Deref;
use std::ops::DerefMut;
use crossbeam_epoch::Owned;
use crossbeam_epoch::Shared;
use crossbeam_epoch::Guard;

static TXN_WRITE_TIME: AtomicU64 = AtomicU64::new(0);

fn get_and_advance_txn_timestamp_acquire_release() -> u64 {
    let old_counter_value = TXN_WRITE_TIME.fetch_add(1, AcqRel);
    old_counter_value + 1
}

// A magic number indicating that the transaction counter is in the middle of
// updating.
const TXN_COUNTER_UPDATING_VAL: u64 = MAX;
// A magic number indicating that this version does not exist at a canonical
// time. This is the timestamp value on shadow versions.
const TXN_COUNTER_NON_CANON: u64 = TXN_COUNTER_UPDATING_VAL - 1;

struct SendSyncPointerWrapper<PointeeType> {
    ptr: * mut PointeeType
}

pub trait FlexibleArrayHeader {
    fn get_flexible_array_len(&self) -> usize;
}

#[repr(C)]
pub struct FlexibleArray<Header: FlexibleArrayHeader, ArrayMember> {
    header: Header,
    flexible_array: [ArrayMember; 0]
}

impl<Header: FlexibleArrayHeader, ArrayMember>
    FlexibleArray<Header, ArrayMember>
{
    fn get_flexible_array_slice_ref(&self) -> &[ArrayMember] {
        unsafe {
            self.get_flexible_array_slice_using_override_len
                (self.header.get_flexible_array_len())
        }
    }

    unsafe fn get_flexible_array_slice_using_override_len
        (&self, override_len: usize) -> &[ArrayMember]
    {
        let ptr_to_flexible_array =
            &self.flexible_array as * const ArrayMember;
        from_raw_parts(ptr_to_flexible_array, override_len)
    }

    unsafe fn get_flexible_array_slice_mut_using_override_len
        (&mut self, override_len: usize) -> &mut [ArrayMember]
    {
        let const_ptr_to_flexible_array =
            &self.flexible_array as * const ArrayMember;
        let mut_ptr_to_flexible_array: * mut ArrayMember =
            transmute(const_ptr_to_flexible_array);
        from_raw_parts_mut(mut_ptr_to_flexible_array, override_len)
    }

    fn get_layout(flexible_array_len: usize) -> Layout {
        let header_layout = Layout::new::<Header>();
        let array_layout =
            Layout::array::<ArrayMember>(flexible_array_len).unwrap();
        header_layout.extend(array_layout).unwrap().0.pad_to_align()
    }

    unsafe fn drop_trailing_array(&mut self, flexible_array_len: usize) {
        let flex_array_mut_slice =
            self.get_flexible_array_slice_mut_using_override_len
                (flexible_array_len);
        for flex_array_member in flex_array_mut_slice {
            drop_in_place(flex_array_member)
        }
    }
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
struct TVarVersionHeader {
    timestamp: AtomicU64,
    next_ptr: * mut TVarVersionHeader,
    // We will often be referring to these inner versions as
    // TVarVersionInner<()> due to losing track of types. This is the type id of
    // the payload, allowing us to check our work to prevent confusion and
    // to automatically look up functions associated with this payload type
    // (for instance, the destructor). We can also use this to derive a layout
    // for the whole version, complete with the possible dynamic array contained
    // within.
    allocator: &'static TVarVersionAllocator,
    payload_type_id: TypeId,
}

impl TVarVersionHeader {

    fn has_payload_type
        <Header: FlexibleArrayHeader + 'static,
         ArrayMember: 'static>(&self) -> bool
    {
        self.payload_type_id ==
            TypeId::of::<FlexibleArray<Header, ArrayMember>>()
    }

    fn get_full_version_ptr
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>(&self)
        -> TVarVersionPtr<Header, ArrayMember>
    {
        if self.has_payload_type::<Header, ArrayMember>() {
            let self_ptr = NonNull::from(self);
            let full_ptr = self_ptr.cast::<TVarVersion<Header, ArrayMember>>();
            TVarVersionPtr {
                0: full_ptr
            }
        } else {
            panic!("Payload did not match expected type!")
        }
    }
}

// This is the above paired with a payload. We can point to the payload without
// any type information and then dynamic cast to this type to get access to the
// payload.
#[repr(C)]
struct TVarVersion<Header: FlexibleArrayHeader, ArrayMember> {
    header: TVarVersionHeader,
    payload: FlexibleArray<Header, ArrayMember>
}

unsafe fn clean_up_tvar_version
    <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>
    (untyped_version: TVarVersionHeaderPtr,
     // For versions that are still mutable, the user could have messed up the
     // header and had it return the wrong size. In this case, the deallocated
     // shadow version will pass in the known size. For immutable versions, the
     // header can be relied upon and we pass in None.
     flexible_array_len_override: Option<usize>)
{
    let mut typed_version =
        untyped_version.cast_to_version_pointer::<Header, ArrayMember>();
    unsafe {
        let payload_ref = &mut typed_version.0.as_mut().payload;
        drop_in_place(&mut payload_ref.header);
        payload_ref.drop_trailing_array
            (flexible_array_len_override
                .unwrap_or_else(|| {
                    payload_ref.header.get_flexible_array_len()
                }));
    };
}

fn dealloc_tvar_version
    <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>
    (untyped_version: TVarVersionHeaderPtr)
{
    let typed_version =
        untyped_version.cast_to_version_pointer::<Header, ArrayMember>();
    let guard = crossbeam_epoch::pin();
    unsafe {
        guard.defer_destroy
            (Shared::from(typed_version.0.as_ptr()
                as * const TVarVersionPtr<Header, ArrayMember>));
    }
}

impl<Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
    TVarVersion<Header, ArrayMember>
{
    fn get_layout(flexible_array_len: usize) -> Layout {
        let payload_layout =
            FlexibleArray::<Header, ArrayMember>::get_layout
                (flexible_array_len);
        let tvar_version_header_layout = Layout::new::<TVarVersionHeader>();
        tvar_version_header_layout
            .extend(payload_layout).unwrap().0.pad_to_align()
    }

    fn initialize_except_trailing_array
        (maybe_uninit_header_mut:
            & mut MaybeUninit<TVarVersion<Header, ArrayMember>>,
         header_init: Header,
         flexible_array_len: usize)
            -> &mut TVarVersion<Header, MaybeUninit<ArrayMember>>
    {
        let layout =
            TVarVersion::<Header, ArrayMember>::get_layout(flexible_array_len);
        let version_for_layout =
            get_or_create_version_allocator_for_layout(layout);
        let tvar_version_header = TVarVersionHeader {
            timestamp: AtomicU64::new(TXN_COUNTER_NON_CANON),
            next_ptr: null_mut(),
            allocator: version_for_layout,
            payload_type_id:
                idempotent_add_version_payload_type::<Header, ArrayMember>(),
        };

        let version_mut =
            maybe_uninit_header_mut.write(TVarVersion {
                header: tvar_version_header,
                payload: FlexibleArray {
                    header: header_init,
                    flexible_array: []
                }
            });
        let raw_version_pointer =
            version_mut as * mut TVarVersion<Header, ArrayMember>;
        let raw_version_pointer_with_uninit_array =
            raw_version_pointer
                .cast::<TVarVersion<Header, MaybeUninit<ArrayMember>>>();
        unsafe { raw_version_pointer_with_uninit_array.as_mut() }.unwrap()
    }

    unsafe fn initialize
        <'a, FlexArrayInit: FnOnce(&mut[MaybeUninit<ArrayMember>])>
        (maybe_uninit_header_mut:
            &'a mut MaybeUninit<TVarVersion<Header, ArrayMember>>,
         header_init: Header,
         flex_array_length: usize,
         flex_array_init_fn: FlexArrayInit)
            -> TVarVersionPtr<Header, ArrayMember>
    {
        let partially_init_version =
            Self::initialize_except_trailing_array
                (maybe_uninit_header_mut, header_init, flex_array_length);
        let uninit_flexible_array_slice =
            partially_init_version.payload.get_flexible_array_slice_mut_using_override_len(flex_array_length);
        flex_array_init_fn(uninit_flexible_array_slice);
        let init_cast_version =
            NonNull::from(partially_init_version)
                .cast::<TVarVersion<Header, ArrayMember>>();
        TVarVersionPtr {
            0: init_cast_version
        }
    }

    // A derived version is a version that consitutes the next time slice of an
    // object. It does not need to be an exact copy; it may expand or shrink the
    // trailing array. It is expected that this transform occurs in two steps:
    // first, the new header is prepared and second, the slice of data after the
    // header is filled in. This is unsafe because the flexible array xform may
    // fail to fill in the entire new trailing array, which would lead to
    // misbehavior.
    unsafe fn alloc
        <FlexibleArrayXformFn: FnOnce(&mut [MaybeUninit<ArrayMember>])>
        (header_init: Header,
         trailing_array_size: usize,
         flexible_array_xform_fn: FlexibleArrayXformFn)
         -> TVarVersionPtr<Header, ArrayMember>
    {
        idempotent_add_version_payload_type::<Header, ArrayMember>();
        let allocated_tvar_version_u8_ptr =
            System.alloc(Self::get_layout(trailing_array_size));
        let allocated_tvar_version_ptr =
            allocated_tvar_version_u8_ptr
                as * mut MaybeUninit<TVarVersion<Header, ArrayMember>>;
        let uninit_version_mut = allocated_tvar_version_ptr.as_mut().unwrap();
        Self::initialize
            (uninit_version_mut,
             header_init,
             trailing_array_size,
             flexible_array_xform_fn)
    }

}

#[derive(Clone, Copy)]
struct TypeInfo {
    layout: Layout,
    dtor_fn: unsafe fn(TVarVersionHeaderPtr, Option<usize>)
}

lazy_static! {
    static ref GLOBAL_TYPE_ID_TO_INFO_MAP:
        Mutex<HashMap<TypeId, TypeInfo>> = Mutex::new(HashMap::new());
}

thread_local! {
    static THREAD_LOCAL_TYPE_ID_TO_INFO_MAP: RefCell<HashMap<TypeId, TypeInfo>>
        = RefCell::new(HashMap::new());
}

fn idempotent_add_version_payload_type
    <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>()
    -> TypeId
{
    use std::collections::hash_map::Entry;
    let this_type_typeid = TypeId::of::<FlexibleArray<Header, ArrayMember>>();

    let new_type_info = TypeInfo {
        layout: Layout::new::<FlexibleArray<Header, ArrayMember>>(),
        dtor_fn: clean_up_tvar_version::<Header, ArrayMember>
    };

    let new_entry_inserted =
        THREAD_LOCAL_TYPE_ID_TO_INFO_MAP.with(|type_id_to_info_map_cell| {
            let mut type_id_to_info_map = type_id_to_info_map_cell.borrow_mut();
            let thread_local_entry =
                type_id_to_info_map.entry(this_type_typeid);
            match thread_local_entry {
                Entry::Occupied(_) => false,
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(new_type_info);
                    true
                }
            }
        });

    if !new_entry_inserted {
        return this_type_typeid;
    }

    // If this was a vacant entry, we have to possibly update the global map
    // as well.
    let mut global_map_lock =
        GLOBAL_TYPE_ID_TO_INFO_MAP.lock().unwrap();
    let global_entry = global_map_lock.entry(this_type_typeid);
    match global_entry {
        Entry::Occupied(_) => { return  this_type_typeid; }
        Entry::Vacant(vacant_entry) => {
            vacant_entry.insert(new_type_info);
            return this_type_typeid;
        }
    }
}

fn fetch_info_for_type_id(type_id: TypeId) -> TypeInfo {
    let optional_thread_local_type_info =
        THREAD_LOCAL_TYPE_ID_TO_INFO_MAP.with(|type_id_to_info_map_cell| {
            let type_id_to_info_map = type_id_to_info_map_cell.borrow();
            type_id_to_info_map.get(&type_id).copied()
        });
    // If we were able to retrieve the info from the thread local map, call it
    // done. Otherwise, we have to look in the global map and update the
    // thread-local map.
    match optional_thread_local_type_info {
        Some(present_entry) => return present_entry,
        None => { }
    };

    let global_type_map_lock = GLOBAL_TYPE_ID_TO_INFO_MAP.lock().unwrap();

    let optional_global_type_info = global_type_map_lock.get(&type_id).copied();

    let global_type_info =
        match optional_global_type_info {
            Some(present_entry) => present_entry,
            None => {
                panic!("TypeID did not have an entry in the global map")
            }
        };

    THREAD_LOCAL_TYPE_ID_TO_INFO_MAP.with(|type_id_to_info_map_cell|{
        type_id_to_info_map_cell.borrow_mut().insert(type_id, global_type_info);
    });
    global_type_info
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct TVarVersionHeaderPtr(NonNull<TVarVersionHeader>);

impl TVarVersionHeaderPtr {

    fn cast_to_version_pointer
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>(&self)
        -> TVarVersionPtr<Header, ArrayMember>
    {
        unsafe { self.0.as_ref() }.get_full_version_ptr()
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

unsafe impl Send for TVarVersionHeaderPtr { }
unsafe impl Sync for TVarVersionHeaderPtr { }

#[derive(PartialEq, Eq)]
struct TVarVersionPtr<Header: FlexibleArrayHeader, ArrayMember>
    (NonNull<TVarVersion<Header, ArrayMember>>);

impl<Header: FlexibleArrayHeader, ArrayMember>
Clone for TVarVersionPtr<Header, ArrayMember> {
    fn clone(&self) -> Self {
        TVarVersionPtr {
            0: self.0
        }
    }
}

impl<Header: FlexibleArrayHeader, ArrayMember>
Copy for TVarVersionPtr<Header, ArrayMember> { }

impl<Header: FlexibleArrayHeader, ArrayMember>
    TVarVersionPtr<Header, ArrayMember>
{
    fn get_header_pointer(&self) -> TVarVersionHeaderPtr {
        TVarVersionHeaderPtr {
            0: NonNull::from(&unsafe { self.0.as_ref() }.header)
        }
    }
}

// An allocator for caching TVar versions in such a fashion that they can be
// re-allocated quickly without synchronization.
struct TVarVersionAllocator {
    layout: Layout,
    // This is the list of objects that are free and ready to serve the layout
    // indicated.
    free_list: AtomicPtr<TVarVersionHeader>,
}

impl TVarVersionAllocator {
    fn new(layout: Layout) -> TVarVersionAllocator {
        TVarVersionAllocator {
            layout,
            free_list: AtomicPtr::new(null_mut())
        }
    }

    fn alloc_shadow_version
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static,
         FlexArrayInitFn: FnOnce(&mut [MaybeUninit<ArrayMember>])>
        (&self, header_init: Header, array_member_init: FlexArrayInitFn)
        -> TVarShadowVersion<Header, ArrayMember>
    {
        let new_size = header_init.get_flexible_array_len();
        assert!(self.layout.size() >=
            size_of::<TVarVersion<Header, ArrayMember>>());
        assert!(self.layout.align() >=
            align_of::<TVarVersion<Header, ArrayMember>>());

        let mut free_list_head_ptr = self.free_list.load(Acquire);
        let tvar_ptr = loop {
            // If the head is equal to null, allocate a new item.
            if free_list_head_ptr == null_mut() {
                break
                    unsafe {
                        TVarVersion::alloc
                            (header_init, new_size, array_member_init)
                    };
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
                            as
                            * mut MaybeUninit<TVarVersion<Header, ArrayMember>>;
                    break
                        unsafe {
                            TVarVersion::<Header, ArrayMember>::initialize
                                (uninit_ptr.as_mut().unwrap(),
                                 header_init,
                                 new_size,
                                 array_member_init)
                        }
                },
                Err(seen_head) => {
                    free_list_head_ptr = seen_head;
                }
            }
        };
        TVarShadowVersion {
            version_ptr: tvar_ptr,
            trailing_array_size: new_size
        }
    }

    fn alloc_duplicate_shadow_version
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static>
        (&self, original: &TVarVersion<Header, ArrayMember>)
        -> TVarShadowVersion<Header, ArrayMember>
    {
        let new_header = original.payload.header.clone();
        self.alloc_shadow_version_copy_prefix
            (original.payload.get_flexible_array_slice_ref(), new_header, None)
    }

    // This variant allocates a new shadow version with the prefix of the
    // flexible array copied from the old value. If the new array is larger than
    // the old array, copy the old array to the front of the new array. If the
    // new array is smaller than the old array, copy only the elements that
    // would be in-range in the new array from the old array.
    fn alloc_shadow_version_copy_prefix
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static>
        (&self,
         original_trailing_slice: &[ArrayMember],
         new_header: Header,
         new_entries_init: Option<ArrayMember>)
         -> TVarShadowVersion<Header, ArrayMember>
    {
        self.alloc_shadow_version
            (new_header,
            |new_slice| {
                for (idx, new_entry_slot) in new_slice.iter_mut().enumerate() {
                    if idx < original_trailing_slice.len() {
                        let entry_init =
                            if idx < original_trailing_slice.len() {
                                original_trailing_slice[idx].clone()
                              } else {
                                  new_entries_init.as_ref().unwrap().clone()
                              };
                        new_entry_slot.write(entry_init);
                    }
                }
            })
    }

    fn return_stale_version_pointer
        (&self,
         stale_ptr: TVarVersionHeaderPtr,
         flexible_array_len_override: Option<usize>)
    {
        let type_id = unsafe { stale_ptr.0.as_ref().payload_type_id };
        let dtor_fn = fetch_info_for_type_id(type_id).dtor_fn;
        unsafe { dtor_fn(stale_ptr, flexible_array_len_override) };

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
        (&self, returned: TVarVersionHeaderPtr)
    {
        // For a formerly canon version, we have to wait until all threads
        // currently running transactions have finished possibly using this
        // version before it can be reused.
        let guard = crossbeam_epoch::pin();
        let alloc_ptr = SendSyncPointerWrapper {
            ptr: NonNull::from(self).as_ptr()
        };
        guard.defer(move || {
            unsafe { alloc_ptr.ptr.as_ref() }
                .unwrap()
                .return_stale_version_pointer(returned, None);
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
            HashMap<HashableLayout, &'static TVarVersionAllocator>> =
                Mutex::new(HashMap::new());
}

thread_local! {
    static THREAD_LAYOUT_TO_ALLOC_MAP:
        RefCell<HashMap<HashableLayout, &'static TVarVersionAllocator>> =
            RefCell::new(HashMap::new());
}

fn get_or_create_version_allocator_for_layout(layout: Layout)
    -> &'static TVarVersionAllocator
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
    let result_ref = match layout_entry {
        Entry::Occupied(present_entry) => {
            present_entry.get()
        },
        Entry::Vacant(vacant_entry) => {
            let entry_ref =
                Box::leak(Box::new(TVarVersionAllocator::new(layout)));
            let insert_result = vacant_entry.insert(entry_ref);
            *insert_result
        }
    };

    THREAD_LAYOUT_TO_ALLOC_MAP.with(|alloc_map_key| {
        alloc_map_key.borrow_mut().insert(hashable_layout, result_ref);
    });
    result_ref
}

// A TVarShadowVersion represents a version of the object that is local to a
// transaction and which is still mutable. If the transaction fails and
// restarts, then this version will be dropped. If the transaction succeeds,
// then this will be converted to a TVarImmVersion so that it can be shared
// between threads without worrying about mutation.
//
// Note: we track the trailing array size on the shadow version because the user
// could possibly change the size and mess up our trailing array tracking,
// leading to memory unsafety. This is no longer an issue if/when this becomes
// an immutable canonical version.
#[derive(Clone, Copy)]
struct TVarShadowVersion<Header: FlexibleArrayHeader, ArrayMember> {
    version_ptr: TVarVersionPtr<Header, ArrayMember>,
    trailing_array_size: usize
}

impl<Header: FlexibleArrayHeader, ArrayMember>
TVarShadowVersion<Header, ArrayMember> {
    fn erase_type(&self) -> TVarShadowVersionTypeErased {
        TVarShadowVersionTypeErased {
            version_ptr: self.version_ptr.get_header_pointer(),
            trailing_array_size: self.trailing_array_size
        }
    }
}

#[derive(Clone, Copy)]
struct TVarShadowVersionTypeErased {
    version_ptr: TVarVersionHeaderPtr,
    trailing_array_size: usize
}

impl TVarShadowVersionTypeErased {
    fn cast_to_shadow_version
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>
        (&self)
        -> TVarShadowVersion<Header, ArrayMember>
    {
        let version_ptr =
            self.version_ptr.cast_to_version_pointer::<Header, ArrayMember>();

        TVarShadowVersion {
            version_ptr,
            trailing_array_size: self.trailing_array_size
        }
    }

    fn return_to_allocator(&self) {
        let version_header_ref = unsafe { self.version_ptr.0.as_ref() };
        version_header_ref.allocator.return_stale_version_pointer
            (self.version_ptr, Some(self.trailing_array_size));
    }
}

#[derive(Copy, Clone)]
struct VersionedTVarTypeErasedRef {
    tvar_ptr: SendSyncPointerWrapper<VersionedTVarTypeErased>
}

struct VersionedTVarRef<Header: FlexibleArrayHeader, ArrayMember> {
    tvar_ref: VersionedTVarTypeErasedRef,
    phantom_header: PhantomData<Header>,
    phantom_array_member: PhantomData<ArrayMember>
}

impl<Header: FlexibleArrayHeader, ArrayMember>
Clone for VersionedTVarRef<Header, ArrayMember> {
    fn clone(&self) -> Self {
        VersionedTVarRef {
            tvar_ref: self.tvar_ref,
            phantom_header: PhantomData,
            phantom_array_member: PhantomData
        }
    }
}

impl<Header: FlexibleArrayHeader, ArrayMember> Copy for
    VersionedTVarRef<Header, ArrayMember> { }

// This is a ref which owns the tvar version contained within. Once all such
// refs are dropped, the tvar is freed.
pub struct SharedTVarRef<Header: FlexibleArrayHeader, ArrayMember> {
    tvar_ref: VersionedTVarRef<Header, ArrayMember>
}

impl<Header: FlexibleArrayHeader, ArrayMember> Drop for
    SharedTVarRef<Header, ArrayMember>
{
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

impl<Header: FlexibleArrayHeader, ArrayMember>
Clone for SharedTVarRef<Header, ArrayMember> {
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

pub type SharedFixedSizeTVarRef<GuardedType> =
    SharedTVarRef<SingletonHeader<GuardedType>, ()>;

struct CanonPtrAndWriteReserved {
    // If this is None, then the tvar is uninit.
    canon_ptr: Option<TVarVersionHeaderPtr>,
    write_reserved: bool
}

impl CanonPtrAndWriteReserved {

    const fn get_canon_ptr_as_int(opt_header_ptr: Option<TVarVersionHeaderPtr>)
        -> u64
    {
        match opt_header_ptr {
            Some(header_ptr) => unsafe { header_ptr.0.as_ptr() as u64 },
            None => 0
        }
    }

    const fn new(opt_canon_pointer: Option<TVarVersionHeaderPtr>,
           write_reserved: bool)
        -> CanonPtrAndWriteReserved
    {
        let canon_ptr_as_int = Self::get_canon_ptr_as_int(opt_canon_pointer);
        assert!(canon_ptr_as_int & 1 == 0);

        CanonPtrAndWriteReserved {
            canon_ptr: opt_canon_pointer,
            write_reserved
        }
    }

    const fn pack(&self) -> u64 {
        let canon_ptr_as_int = Self::get_canon_ptr_as_int(self.canon_ptr);
        // Assert that the bottom bit of the pointer is 0.
        assert!(canon_ptr_as_int & 1 == 0);
        canon_ptr_as_int | if self.write_reserved { 1 } else { 0 }
    }

    fn unpack(packed: u64) -> CanonPtrAndWriteReserved {
        let write_reserved = (packed & 1) == 1;
        let canon_raw = ((packed >> 1) << 1) as * mut TVarVersionHeader;
        let opt_canon_nonnull = NonNull::new(canon_raw);
        let opt_canon_ptr =
            opt_canon_nonnull.map(|canon_nonnull| {
                TVarVersionHeaderPtr { 0: canon_nonnull }
            });
        CanonPtrAndWriteReserved::new(opt_canon_ptr, write_reserved)
    }
}

const NON_REFCT_TVAR: u64 = MAX;

const RETIRED_CANON_OBJECT: u8 = 0;
const RETIRED_CANON_PTR: * const u8 = &RETIRED_CANON_OBJECT as * const u8;

struct VersionedTVarTypeErased {
    packed_canon_ptr: AtomicU64,
    refct: AtomicU64
}

impl VersionedTVarTypeErased {
    fn fetch_and_unpack_canon_ptr(&self, ordering: Ordering)
            -> CanonPtrAndWriteReserved {
        let packed_canon_ptr = self.packed_canon_ptr.load(ordering);
        CanonPtrAndWriteReserved::unpack(packed_canon_ptr)
    }

    fn has_canon_version_ptr(&self, candidate: Option<TVarVersionHeaderPtr>)
        -> bool
    {
        self.fetch_and_unpack_canon_ptr(Relaxed).canon_ptr == candidate
    }

    fn get_current_canon_version(&self) -> Option<TVarVersionHeaderPtr> {
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

        former_canon_ptr.and_then(|present_former_canon_ptr| {
            let former_canon_ptr_allocator =
                unsafe { present_former_canon_ptr.0.as_ref() }.allocator;
            former_canon_ptr_allocator
                .return_formerly_canon_pointer(present_former_canon_ptr);
            Some(())
        });
    }
}

// This is a flexible array header that has no trailing array; all of the
// information is in the header itself. This is the common case; only advanced
// users building growable data structures will probably want the full flexible
// array functionality.
pub struct SingletonHeader<GuardedType: Clone>(GuardedType);

impl<GuardedType: Clone> Clone
    for SingletonHeader<GuardedType>
{
    fn clone(&self) -> Self {
        Self {
            0: self.0.clone()
        }
    }
}

impl<GuardedType: Clone> SingletonHeader<GuardedType> {
    fn new(init: GuardedType) -> SingletonHeader<GuardedType> {
        SingletonHeader {
            0: init
        }
    }
}

impl<GuardedType: Clone> FlexibleArrayHeader
    for SingletonHeader<GuardedType>
{
    fn get_flexible_array_len(&self) -> usize {
        0
    }
}

pub struct VersionedTVar<Header: FlexibleArrayHeader, ArrayMember> {
    inner_type_erased: VersionedTVarTypeErased,
    phantom_header: PhantomData<Header>,
    phantom_array_member: PhantomData<ArrayMember>
}

unsafe impl<Header: FlexibleArrayHeader, ArrayMember>
Send for VersionedTVar<Header, ArrayMember> { }

impl<Header : FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
VersionedTVar<Header, ArrayMember> {

    pub const fn new_empty() -> VersionedTVar<Header, ArrayMember> {
        VersionedTVar {
            inner_type_erased: VersionedTVarTypeErased {
                refct: AtomicU64::new(NON_REFCT_TVAR),
                packed_canon_ptr:
                    AtomicU64::new
                        (CanonPtrAndWriteReserved::new(None, false).pack())
            },
            phantom_header: PhantomData,
            phantom_array_member: PhantomData
        }
    }

    pub fn new(header_init: Header, trailing_array_init: &[ArrayMember])
            -> VersionedTVar<Header, ArrayMember>
    {
        let flexible_array_len = header_init.get_flexible_array_len();
        assert_eq!
            (flexible_array_len, trailing_array_init.len());
        // To use the same flow of functions that a regular update to the TVar
        // uses, just create a shadow version and turn it into a TVarImmVersion.

        let allocator_ref =
            get_or_create_version_allocator_for_layout
                (TVarVersion::<Header, ArrayMember>::get_layout
                    (flexible_array_len));

        let init_shadow_version =
            allocator_ref.alloc_shadow_version
                (header_init,
                 |new_slice: &mut [MaybeUninit<ArrayMember>]| {
                    for (original_item, uninit_item) in
                        trailing_array_init.iter().zip(new_slice.iter_mut())
                    {
                        uninit_item.write(original_item.clone());
                    }
                 });

        let shadow_version_ptr = init_shadow_version.version_ptr;
        let mut shadow_version_nonnull_ptr = shadow_version_ptr.0;

        let shadow_version_mut = unsafe { shadow_version_nonnull_ptr.as_mut() };
        let shadow_version_header = &mut shadow_version_mut.header;

        let canon_ptr_and_write_reserved =
            CanonPtrAndWriteReserved::new
                (Some(shadow_version_ptr.get_header_pointer()), false);

        // If lazy_static! is used to create a VersionedTVar, the initialization
        // of this tvar can occur at an arbitrary time - possibly after other
        // transactions have launched. To prevent writes performed by new from
        // being lost, take up a version for the initialization.
        let this_txn_time_number =
            get_and_advance_txn_timestamp_acquire_release();

        shadow_version_header.timestamp.store(this_txn_time_number, Release);
        let result: VersionedTVar<Header, ArrayMember> =
            VersionedTVar {
                inner_type_erased: VersionedTVarTypeErased {
                    refct: AtomicU64::new(NON_REFCT_TVAR),
                    packed_canon_ptr:
                        AtomicU64::new(canon_ptr_and_write_reserved.pack())
                },
                phantom_header: PhantomData,
                phantom_array_member: PhantomData
            };
        result
    }

    fn new_shared_inner(shared_tvar: VersionedTVar<Header, ArrayMember>)
        -> SharedTVarRef<Header, ArrayMember>
    {
        let owned_ptr = Owned::new(shared_tvar);
        // Initially, the refct indicates that this is a non-refcounted tvar.
        // Update it to make it have a refcount of 1.
        (*owned_ptr).inner_type_erased.refct.store(1, Relaxed);
        let guard = crossbeam_epoch::pin();
        let shared_ptr = owned_ptr.into_shared(&guard);
        let raw_tvar_ptr = shared_ptr.as_raw();
        let raw_mut_tvar_ptr =
            raw_tvar_ptr as * mut VersionedTVar<Header, ArrayMember>;
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
                phantom_header: PhantomData,
                phantom_array_member: PhantomData
            }
        }

    }

    pub fn new_shared(header_init: Header, trailing_array_init: &[ArrayMember])
            -> SharedTVarRef<Header, ArrayMember>
    {
        Self::new_shared_inner
            (VersionedTVar::new(header_init, trailing_array_init))
    }

    pub fn new_shared_empty() -> SharedTVarRef<Header, ArrayMember> {
        Self::new_shared_inner
            (VersionedTVar::<Header, ArrayMember>::new_empty())
    }

    pub fn retire(&self) {
        self.inner_type_erased.retire();
    }
}

type FixedSizeTVar<GuardedType> =
    VersionedTVar<SingletonHeader<GuardedType>, ()>;

impl<GuardedType: Clone + 'static> FixedSizeTVar<GuardedType> {
    pub fn new_fixed_size(init: GuardedType) -> FixedSizeTVar<GuardedType> {
        let empty_unit_array: [(); 0] = [ ];
        Self::new(SingletonHeader::new(init), &empty_unit_array)
    }

    pub fn new_shared_fixed_size(init: GuardedType)
        -> SharedFixedSizeTVarRef<GuardedType> {
        let empty_unit_array: [(); 0] = [ ];
        Self::new_shared(SingletonHeader::new(init), &empty_unit_array)
    }
}

struct CapturedTVarCacheEntry {
    captured_version_ptr: Option<TVarVersionHeaderPtr>,
    shadow_copy_ptr: Option<TVarShadowVersionTypeErased>,
    in_use: bool
}

impl CapturedTVarCacheEntry {

    fn new<Header: FlexibleArrayHeader + Clone + 'static,
           ArrayMember: Clone + 'static>
        (opt_captured_version: Option<TVarVersionPtr<Header, ArrayMember>>)
        -> CapturedTVarCacheEntry
    {
        let opt_captured_version_header_ptr =
            opt_captured_version.map
                (|captured_version|{ captured_version.get_header_pointer() });
        CapturedTVarCacheEntry {
            captured_version_ptr: opt_captured_version_header_ptr,
            shadow_copy_ptr: None,
            in_use: false,
        }
    }

    fn is_filled(&self) -> bool {
        self.captured_version_ptr.is_some() || self.shadow_copy_ptr.is_some()
    }

    fn is_empty(&self) -> bool {
        !self.is_filled()
    }

    fn assert_empty_and_fill
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static>
        (&mut self,
         header_init: Header,
         array_member_init: &[ArrayMember])
    {
        assert!
            (self.is_empty(),
             "CapturedTVar was expected to be empty, but was filled");
        let flexible_array_len = header_init.get_flexible_array_len();
        assert_eq!
            (flexible_array_len,
             array_member_init.len(),
             "Expected the header and the array member inits to agree on \
              flexible array length");
        let allocator_ref =
            get_or_create_version_allocator_for_layout
                (TVarVersion::<Header, ArrayMember>::get_layout
                    (flexible_array_len));
        let shadow_version =
            allocator_ref.alloc_shadow_version
                (header_init, move |maybe_uninit_slice|{
                    for (uninit_array_member, init_array_member) in
                        maybe_uninit_slice
                            .iter_mut()
                            .zip(array_member_init.iter())
                    {
                        uninit_array_member.write(init_array_member.clone());
                    }
                });
        self.shadow_copy_ptr = Some(shadow_version.erase_type());
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state| {
            *(per_thread_txn_state.borrow().is_write_txn.borrow_mut()) = true;
        });
    }

    fn get_flexible_array_len
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>(&self)
        -> usize
    {
        match self.shadow_copy_ptr {
            Some(present_shadow_copy) => {
                present_shadow_copy.trailing_array_size
            },
            None => {
                let opt_captured_version_ptr =
                    self.get_captured_version::<Header, ArrayMember>();
                match opt_captured_version_ptr {
                    Some(captured_version_ptr) => {
                        let captured_version_ref =
                            unsafe { captured_version_ptr.0.as_ref() };
                        captured_version_ref
                            .payload.header.get_flexible_array_len()
                    },
                    None =>
                        panic!("Attempt to get flexible array len on an \
                                unfilled tvar")
                }
            }
        }
    }

    fn get_captured_version
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>(&self)
         -> Option<TVarVersionPtr<Header, ArrayMember>>
    {
        self.captured_version_ptr.map(|present_captured_version_ptr| {
            present_captured_version_ptr
                .cast_to_version_pointer::<Header, ArrayMember>()
        })
    }

    fn get_optional_shadow_copy
        <Header: FlexibleArrayHeader + 'static, ArrayMember: 'static>(&self)
        -> Option<TVarShadowVersion<Header, ArrayMember>>
    {
        match self.shadow_copy_ptr {
            Some(present_shadow_copy_ptr) => {
                Some(present_shadow_copy_ptr
                    .cast_to_shadow_version::<Header, ArrayMember>())
            },
            None => None
        }
    }

    fn get_shadow_copy_create_if_not_present
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static>(&mut self)
            -> TVarShadowVersion<Header, ArrayMember>
    {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state| {
            *(per_thread_txn_state.borrow().is_write_txn.borrow_mut()) = true;
        });

        match self.get_optional_shadow_copy() {
            Some(already_present_shadow_copy) => already_present_shadow_copy,
            None => {
                let opt_captured_version_ptr =
                    self.get_captured_version::<Header, ArrayMember>();
                match opt_captured_version_ptr {
                    Some(captured_version_ptr) => {
                        let captured_version_ref =
                            unsafe { captured_version_ptr.0.as_ref() };
                        let allocator_ref =
                            captured_version_ref.header.allocator;
                        let dup_shadow_version =
                            allocator_ref
                                .alloc_duplicate_shadow_version
                                    (captured_version_ref);
                        self.shadow_copy_ptr =
                            Some(dup_shadow_version.erase_type());
                        dup_shadow_version
                    },
                    None => {
                        panic!("Attempt to duplicate the version of \
                                an unfilled tvar")
                    }
                }
            }
        }
    }

    fn get_working_version_ptr<'tvar_ref,
              Header: FlexibleArrayHeader + 'static,
              ArrayMember: 'static>(&self)
                -> TVarVersionPtr<Header, ArrayMember>
    {
        let opt_shadow_copy_ref =
            self.get_optional_shadow_copy::<Header, ArrayMember>();
        match opt_shadow_copy_ref {
            Some(present_shadow_copy) => present_shadow_copy.version_ptr,
            None => match self.get_captured_version::<Header, ArrayMember>() {
                Some(captured_version) => captured_version,
                None => panic!
                    ("Attempt to get the captured version of an unfilled \
                      tvar")
            }
        }
    }

    pub fn get_captured_tvar
        <'tvar_ref,
            Header: FlexibleArrayHeader + Clone + 'static,
            ArrayMember: Clone + 'static>
        (&mut self, tvar_ref: &'tvar_ref VersionedTVarTypeErased)
        -> CapturedTVar<'tvar_ref, Header, ArrayMember>
    {
        if self.in_use {
            panic!("Attempt to capture the same TVar more than once \
                   simultaneously");
        }
        self.in_use = true;
        CapturedTVar {
            tvar_ref,
            phantom_header: PhantomData,
            phantom_array_member: PhantomData
        }
    }

    pub fn resize_copy_prefix
        <Header: FlexibleArrayHeader + Clone + 'static,
         ArrayMember: Clone + 'static>
    (&mut self, new_header: Header, new_entries_init: Option<ArrayMember>) {
        PER_THREAD_TXN_STATE.with(|per_thread_txn_state| {
            *(per_thread_txn_state.borrow().is_write_txn.borrow_mut()) = true;
        });

        let flexible_array_len = new_header.get_flexible_array_len();
        // If the flexible array is already the same size as the requested size,
        // nothing more to do.
        if flexible_array_len ==
            self.get_flexible_array_len::<Header, ArrayMember>()
        {
            let mut shadow_copy =
                self.get_shadow_copy_create_if_not_present
                    ::<Header, ArrayMember>();
            let version_mut = unsafe { shadow_copy.version_ptr.0.as_mut() };
            version_mut.payload.header = new_header;
        } else {
            // Otherwise, allocate a new shadow version with the provided size.
            let allocator_ref = get_or_create_version_allocator_for_layout
                (TVarVersion::<Header, ArrayMember>::get_layout
                        (flexible_array_len));
            let version_ptr =
                self.get_working_version_ptr::<Header, ArrayMember>();
            let flexible_array_len =
                self.get_flexible_array_len::<Header, ArrayMember>();
            let version_ref = unsafe { version_ptr.0.as_ref() };
            let original_trailing_slice =
                unsafe {
                    version_ref
                        .payload
                        .get_flexible_array_slice_using_override_len
                            (flexible_array_len)
                };
            let new_shadow_version =
                allocator_ref.alloc_shadow_version_copy_prefix
                    (original_trailing_slice,
                    new_header,
                    new_entries_init);
            let old_shadow_version =
                self.shadow_copy_ptr.replace(new_shadow_version.erase_type());
            match old_shadow_version {
                Some(present_old_shadow_version) => {
                    present_old_shadow_version.return_to_allocator()
                },
                None => { },
            }
        }
    }
}

#[derive(Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
struct CapturedTVarIndex(NonNull<VersionedTVarTypeErased>);

impl CapturedTVarIndex {
    fn new(tvar_ref: &VersionedTVarTypeErased) -> CapturedTVarIndex {
        CapturedTVarIndex {
            0: NonNull::from(tvar_ref)
        }
    }
}

pub struct CapturedTVarRef
    <'tvar,
     'captured_tvar,
     'flex_array_ref,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
{
    captured_tvar: &'captured_tvar CapturedTVar<'tvar, Header, ArrayMember>,
    inner_ref: &'flex_array_ref FlexibleArray<Header, ArrayMember>,
    // It is possible that this ref points to an immutable version, in which
    // case we can trust the header to get the size, or a shadow copy, in which
    // case the header is mutable and untrustworthy, requiring us to provide an
    // override. Making this optional would just force us to store an additonal
    // bool, so just always put an override in.
}

impl<'tvar,
     'captured_tvar,
     'flex_array_ref,
     Header: FlexibleArrayHeader + Clone,
     ArrayMember: Clone>
CapturedTVarRef<'tvar, 'captured_tvar, 'flex_array_ref, Header, ArrayMember>
{
    pub fn get_flexible_array_header_ref(&self) -> &Header {
        &self.inner_ref.header
    }

    pub fn get_flexible_array_slice(&self) -> &[ArrayMember] {
        unsafe {
            self.inner_ref
                .get_flexible_array_slice_using_override_len
                    (self.captured_tvar.get_flexible_array_len())
        }
    }
}

// We can always downgrade a mut to a ref, as consuming the unique mut means
// that there are no remaining borrows of the tvar value.
impl<'a, 'b, 'c,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
From<CapturedTVarMut<'a, 'b, 'c, Header, ArrayMember>> for
    CapturedTVarRef<'a, 'b, 'c, Header, ArrayMember>
{
    fn from(captured_tvar_mut: CapturedTVarMut<'a, 'b, 'c, Header, ArrayMember>)
        -> CapturedTVarRef<'a, 'b, 'c, Header, ArrayMember>
    {
        CapturedTVarRef {
            captured_tvar: captured_tvar_mut.captured_tvar,
            inner_ref: captured_tvar_mut.inner_mut
        }
    }
}

type CapturedFixedSizeTVarRef<'a, 'b, 'c, GuardedType> =
    CapturedTVarRef<'a, 'b, 'c, SingletonHeader<GuardedType>, ()>;

impl<'a, 'b, 'c, GuardedType: Clone>
AsRef<GuardedType> for CapturedFixedSizeTVarRef<'a, 'b, 'c, GuardedType>
{
    fn as_ref(&self) -> &GuardedType {
        &self.inner_ref.header.0
    }
}

impl<'a, 'b, 'c, GuardedType: Clone> Deref for
    CapturedFixedSizeTVarRef<'a, 'b, 'c, GuardedType>
{
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        &self.inner_ref.header.0
    }
}

pub struct CapturedTVarMut
    <'tvar,
     'captured_tvar,
     'flex_array_ref,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
{
    captured_tvar: &'captured_tvar mut CapturedTVar<'tvar, Header, ArrayMember>,
    inner_mut: &'flex_array_ref mut FlexibleArray<Header, ArrayMember>
}

impl<'a, 'b, 'c,
     Header: FlexibleArrayHeader + Clone,
     ArrayMember: Clone>
CapturedTVarMut<'a, 'b, 'c, Header, ArrayMember>
{
    pub fn get_flexible_array_header_ref(&self) -> &Header {
        &self.inner_mut.header
    }

    pub fn get_flexible_array_header_mut(&mut self) -> &mut Header {
        &mut self.inner_mut.header
    }

    pub fn get_flexible_array_slice(&self) -> &[ArrayMember] {
        unsafe {
            self.inner_mut
                .get_flexible_array_slice_using_override_len
                    (self.captured_tvar.get_flexible_array_len())
        }
    }

    pub fn get_flexible_array_slice_mut(&mut self) -> &mut [ArrayMember] {
        unsafe {
            self.inner_mut
                .get_flexible_array_slice_mut_using_override_len
                    (self.captured_tvar.get_flexible_array_len())
        }
    }

}

type CapturedFixedSizeTVarMut<'a, 'b, 'c, GuardedType> =
    CapturedTVarMut<'a, 'b, 'c, SingletonHeader<GuardedType>, ()>;

impl<'a, 'b, 'c, GuardedType: Clone>
AsRef<GuardedType> for CapturedFixedSizeTVarMut<'a, 'b, 'c, GuardedType>
{
    fn as_ref(&self) -> &GuardedType {
        &self.inner_mut.header.0
    }
}

impl<'a, 'b, 'c, GuardedType: Clone>
AsMut<GuardedType> for CapturedFixedSizeTVarMut<'a, 'b, 'c, GuardedType> {
    fn as_mut(&mut self) -> &mut GuardedType {
        &mut self.inner_mut.header.0
    }
}

impl<'a, 'b, 'c, GuardedType: Clone>
Deref for CapturedFixedSizeTVarMut<'a, 'b, 'c, GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        &self.inner_mut.header.0
    }
}

impl<'a, 'b, 'c, GuardedType: Clone>
DerefMut for CapturedFixedSizeTVarMut<'a, 'b, 'c, GuardedType> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner_mut.header.0
    }
}

pub struct CapturedTVar
    <'key,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
{
    tvar_ref: &'key VersionedTVarTypeErased,
    phantom_header: PhantomData<Header>,
    phantom_array_member: PhantomData<ArrayMember>
}

impl<'key,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
CapturedTVar<'key, Header, ArrayMember> {

    pub fn is_filled(&self) -> bool {
        self.with_captured_tvar_ref_cell(|tvar_cache_entry| {
            tvar_cache_entry.borrow().is_filled()
        })
    }

    pub fn is_empty(&self) -> bool {
        !self.is_filled()
    }

    pub fn assert_empty_and_fill
        (&mut self,
         header_init: Header,
         array_member_init: &[ArrayMember])
    {
        self.with_captured_tvar_ref_cell(|tvar_cache_entry| {
            tvar_cache_entry.borrow_mut().assert_empty_and_fill
                (header_init, array_member_init);
        });
    }

    pub fn fill_if_empty
        (&mut self,
         header_init: Header,
         array_member_init: &[ArrayMember])
    {
        if self.is_empty() {
            self.assert_empty_and_fill(header_init, array_member_init);
        }
    }

    fn get_cache_index(&self) -> CapturedTVarIndex {
        let tvar_const_ptr = self.tvar_ref as * const VersionedTVarTypeErased;
        CapturedTVarIndex {
            0: NonNull::new(unsafe { transmute(tvar_const_ptr) }).unwrap()
        }
    }

    fn get_flexible_array_len(&self) -> usize {
        self.with_captured_tvar_ref_cell(|tvar_cache_entry| {
            tvar_cache_entry
                .borrow()
                .get_flexible_array_len::<Header, ArrayMember>()
        })
    }

    fn with_captured_tvar_ref_cell
        <ReturnType, TVarFn: FnOnce(&RefCell<CapturedTVarCacheEntry>) -> ReturnType>
        (&self, tvar_fn: TVarFn) -> ReturnType
    {
        PER_THREAD_TXN_STATE.with(|per_thread_state_key| {
            let per_thread_state_ref = per_thread_state_key.borrow();
            let captured_tvar_ref_cell =
                &per_thread_state_ref
                    .captured_tvar_cache[&self.get_cache_index()];
            tvar_fn(captured_tvar_ref_cell)
        })
    }

    pub fn get_captured_tvar_ref(&self) -> CapturedTVarRef<Header, ArrayMember>
    {
        assert!
            (self.is_filled(),
             "Cannot get a tvar ref to an unfilled tvar");
        let inner_ref =
            self.with_captured_tvar_ref_cell(|captured_tvar_cache_entry|{
                let captured_tvar_cache_entry_ref =
                    captured_tvar_cache_entry.borrow();
                let working_version_ptr =
                    captured_tvar_cache_entry_ref
                        .get_working_version_ptr::<Header, ArrayMember>();
                unsafe { &(*working_version_ptr.0.as_ptr()).payload }
            });
        CapturedTVarRef {
            captured_tvar: &self,
            inner_ref
        }
    }

    pub fn get_captured_tvar_mut
        <'captured_tvar, 'flex_array_ref>
    (&'captured_tvar mut self)
            -> CapturedTVarMut
                <'key, 'captured_tvar, 'flex_array_ref, Header, ArrayMember>
    {
        assert!
            (self.is_filled(),
             "Cannot get a mutable tvar ref to an unfilled tvar");
        let shadow_version =
            self.with_captured_tvar_ref_cell(|captured_tvar_cache_entry|{
                let mut captured_tvar_cache_entry_mut =
                    captured_tvar_cache_entry.borrow_mut();
                captured_tvar_cache_entry_mut
                    .get_shadow_copy_create_if_not_present::
                        <Header, ArrayMember>()
            });
        let version_ptr = shadow_version.version_ptr;

        CapturedTVarMut {
            captured_tvar: self,
            inner_mut: unsafe { &mut (*version_ptr.0.as_ptr()).payload }
        }
    }

    pub fn resize_copy_prefix
        <'captured_tvar, 'flex_array_ref>
        (&'captured_tvar mut self,
         header: Header,
         new_entry_fill: Option<ArrayMember>)
        -> CapturedTVarMut
            <'key, 'captured_tvar, 'flex_array_ref, Header, ArrayMember>
    {
        self.with_captured_tvar_ref_cell(|captured_tvar| {
            captured_tvar.borrow_mut().resize_copy_prefix
                (header, new_entry_fill)
        });
        self.get_captured_tvar_mut()
    }
}

impl<'key,
     Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
Drop for CapturedTVar<'key, Header, ArrayMember>
{
    fn drop(&mut self) {
        self.with_captured_tvar_ref_cell(|captured_tvar_cache_entry| {
            let mut cache_entry_mut = captured_tvar_cache_entry.borrow_mut();
            assert!(cache_entry_mut.in_use);
            cache_entry_mut.in_use = false;
        })
    }
}


pub type CapturedFixedSizedTVarCacheKey<'key, GuardedType> =
    CapturedTVar<'key, SingletonHeader<GuardedType>, ()>;

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
        IndexMap<CapturedTVarIndex, RefCell<CapturedTVarCacheEntry>>,
    is_write_txn: RefCell<bool>,
    // Writes at and before this version happened-before this txn, and thus are
    // guaranteed to be consistent. Writes after it, on the other hand, can be
    // skewed, and require a new acquire of the counter.
    txn_version_acquisition_threshold: u64
}

impl PerThreadTransactionState {
    fn reset_txn_state(&mut self, drop_shadow_versions: bool) {
        if drop_shadow_versions {
            for (_, captured_tvar_cell) in
                self.captured_tvar_cache.drain(..)
            {
                let mut mut_captured_tvar = captured_tvar_cell.borrow_mut();
                let previous_shadow_copy =
                    mut_captured_tvar.shadow_copy_ptr.take();
                match previous_shadow_copy {
                    Some(present_shadow_copy) =>
                        present_shadow_copy.return_to_allocator(),
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
enum TVarIndexResult<Header: FlexibleArrayHeader, ArrayMember> {
    KnownFreshIndex,
    MaybeNotFreshContext(Option<TVarVersionPtr<Header, ArrayMember>>, u64)
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
            let tvar_ref = unsafe {&mut (*tvar_ptr.0.as_ptr()) };
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
                let tvar_ref = unsafe {&mut (*tvar_ptr.0.as_ptr()) };
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
            let tvar_ref = unsafe {&mut (*tvar_ptr.0.as_ptr()) };
            match captured_tvar.borrow().shadow_copy_ptr {
                Some(inner_shadow_copy_ptr) => {
                    let shadow_version_header_ptr =
                        inner_shadow_copy_ptr.version_ptr;
                    let raw_shadow_copy_ptr =
                        shadow_version_header_ptr.0.as_ptr();
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
                                (Some(shadow_version_header_ptr), false).pack(),
                             Release)
                },
                None => tvar_ref.clear_write_reservation()
            }
        }

        // At this point, we have updated all of the tvars with our new
        // versions, but none of those versions have a canonical time. Now,
        // we must fetch the canonical time for this transaction. We perform
        // a fetch-add with acquire-release ordering. This means that any
        // Acquire read seeing at least the number we move the counter to
        // must also happen-after all of the swaps we just performed.

        let this_txn_time_number =
            get_and_advance_txn_timestamp_acquire_release();

        // Now that we have completed the commit, we still have to
        // perform some cleanup actions. For all captured tvars where we
        // swapped in a new value, we must place the old canonical version
        // on the stale list.
        for (_, captured_tvar) in captured_tvar_cache_mut.drain(..) {
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
                    .version_ptr
                    .0
                    .as_mut()
                    .timestamp
                    .store(this_txn_time_number, Release);
            }
            // For writes, place the old canon version onto the stale
            // version list.
            let opt_old_canon = captured_tvar.borrow().captured_version_ptr;
            opt_old_canon.and_then(|old_canon|{
                let old_canon_ref = unsafe { old_canon.0.as_ref() };
                old_canon_ref
                    .allocator
                    .return_formerly_canon_pointer(old_canon);
                Some(())
            });
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
                let type_erased_tvar =
                    unsafe { type_erased_tvar_ptr.0.as_ref() };
                if !type_erased_tvar.has_canon_version_ptr
                    (captured_tvar.borrow().captured_version_ptr)
                {
                    return Err(self.make_transaction_error());
                }
            }
            Ok(())
        })
    }

    pub fn capture_tvar_ref
    <Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
        (&self, shared_tvar_ref: &SharedTVarRef<Header, ArrayMember>)
            -> Result<CapturedTVar<Header, ArrayMember>, TxnErrStatus>
    {
        let inner_tvar_ref = &shared_tvar_ref.tvar_ref;
        self.capture_type_erased_tvar
            (unsafe {&mut *inner_tvar_ref.tvar_ref.tvar_ptr.ptr})
    }

    pub fn capture_tvar
    <'a, Header: FlexibleArrayHeader + Clone + 'static,
     ArrayMember: Clone + 'static>
        (&self, tvar: &'a VersionedTVar<Header, ArrayMember>)
            -> Result<CapturedTVar<'a, Header, ArrayMember>,
                      TxnErrStatus>
    {
        self.capture_type_erased_tvar(&tvar.inner_type_erased)
    }


    fn capture_type_erased_tvar
        <'a, Header: FlexibleArrayHeader + Clone + 'static,
        ArrayMember: Clone + 'static>
        (&self, tvar: &'a VersionedTVarTypeErased)
            -> Result<CapturedTVar<'a, Header, ArrayMember>,
                      TxnErrStatus>
    {
        let captured_tvar_index = CapturedTVarIndex::new(tvar);

        let version_acquisition_threshold =
            self.with_per_thread_txn_state_ref(|per_thread_state_ref|{
                per_thread_state_ref.txn_version_acquisition_threshold
            });

        let index_lookup_result = self.with_per_thread_txn_state_mut
            (|per_thread_state_mut| {
                let captured_tvar_entry =
                    per_thread_state_mut
                        .captured_tvar_cache
                        .entry(captured_tvar_index);
                let tvar_index_result: TVarIndexResult<Header, ArrayMember> =
                    match captured_tvar_entry
                    {
                    // If the entry is already there, return the known index.
                    Occupied(_) => {
                        TVarIndexResult::KnownFreshIndex
                    },
                    Vacant(vacant_entry) => {
                        let versioned_tvar_type_erased =
                            unsafe { captured_tvar_index.0.as_ref() };
                        let current_canon =
                            versioned_tvar_type_erased
                                .get_current_canon_version();
                        let cast_current_canon =
                            current_canon.map(|present_current_canon| {
                                present_current_canon
                                    .cast_to_version_pointer
                                    ::<Header, ArrayMember>()
                            });
                        let captured_tvar =
                            CapturedTVarCacheEntry::new(cast_current_canon);
                        // The captured tvar we inserted may have a timestamp
                        // that did not happen before the current transaction
                        // start. If that's the case, we need to check that all
                        // captured entries (except the just captured one) are
                        // still current.
                        let pointee_timestamp = match current_canon {
                            Some(present_current_canon) =>
                                present_current_canon
                                    .get_pointee_timestamp_val_expect_canon(),
                            None => 0
                        };
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
                                (cast_current_canon, pointee_timestamp)
                        }
                    }
                };
                tvar_index_result
            });

        match index_lookup_result {
            TVarIndexResult::KnownFreshIndex => {
                let captured_tvar =
                    self.with_per_thread_txn_state_ref
                        (|per_thread_txn_state_ref| {
                            per_thread_txn_state_ref
                                .captured_tvar_cache
                                .get(&captured_tvar_index)
                                .unwrap()
                                .borrow_mut()
                                .get_captured_tvar(tvar)
                    });
                Ok(captured_tvar)
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

                let mut new_cache_entry =
                    CapturedTVarCacheEntry::new(current_canon);

                let captured_tvar = new_cache_entry.get_captured_tvar(tvar);

                self.with_per_thread_txn_state_mut
                    (|per_thread_txn_state_mut| {
                    let previous_value_expect_none =
                        per_thread_txn_state_mut
                            .captured_tvar_cache
                            .insert
                                (captured_tvar_index,
                                 RefCell::new(new_cache_entry));
                    assert!(previous_value_expect_none.is_none());
                    Ok(captured_tvar)
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
    use std::convert::TryInto;
    use super::*;

    mod test1_state {
        use super::*;
        lazy_static! {
            pub static ref TVAR_INT1: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(5);
            pub static ref TVAR_INT2: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(7);
        }
    }

    // A simple test where one thread alters the state of two tvars and another
    // thread reads that state.

    #[test]
    fn test1_simple_tvar_use() -> Result<(), TxnErrStatus> {
        use test1_state::*;
        let thread1 = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                let mut tvar1_key = txn.capture_tvar(&TVAR_INT1)?;
                let mut tvar2_key = txn.capture_tvar(&TVAR_INT2)?;

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
            pub static ref TVAR_INT1: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(0);
            pub static ref TVAR_INT2: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(0);

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
                let (mut tvar1_key, mut tvar2_key) = {
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

                    *txn.capture_tvar(&TVAR_INT1)?
                        .get_captured_tvar_mut() = 2;
                    *txn.capture_tvar(&TVAR_INT2)?
                        .get_captured_tvar_mut() = 5;
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
            pub static ref TVAR_INT1: FixedSizeTVar<u8> =
                FixedSizeTVar::new_fixed_size(17);
            pub static ref TVAR_INT2: FixedSizeTVar<u8> =
                FixedSizeTVar::new_fixed_size(25);

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

                    *txn.capture_tvar(&TVAR_INT1)?
                        .get_captured_tvar_mut() += 1;
                    *txn.capture_tvar(&TVAR_INT2)?
                        .get_captured_tvar_mut() += 1;
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
        next: SharedFixedSizeTVarRef<Option<TVarLinkedListNode<PayloadType>>>,
        payload: PayloadType
    }

    mod test4_state {
        use super::*;

        lazy_static! {
            pub static ref LIST_HEAD:
                FixedSizeTVar<Option<TVarLinkedListNode<u64>>> =
                    FixedSizeTVar::new_fixed_size(None);
        }
    }

    fn add_number_to_list_inner
        (txn: &VersionedTransaction,
         current_node_ref: &TVarLinkedListNode<u64>,
         num: u64)
            -> Result<(), TxnErrStatus>
    {
        assert!(current_node_ref.payload < num);
        let mut captured_next =
            txn.capture_tvar_ref(&current_node_ref.next)?;

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
                    next: VersionedTVar::new_shared_fixed_size(next_mut.clone())
                });
            return Ok(());
        }
        let next_captured_ref =
            captured_next.get_captured_tvar_ref();

        let next_ref = (*next_captured_ref).as_ref();
        let next_node = next_ref.as_ref().unwrap();
        add_number_to_list_inner(txn, next_node, num)
    }

    fn add_number_to_list(num: u64) {
        use test4_state::*;
        VersionedTransaction::start_txn(|txn| {
            let mut current_node_capture = txn.capture_tvar(&LIST_HEAD)?;

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
                             next: VersionedTVar::new_shared_fixed_size
                                (list_head_mut.clone())
                        });
                return Ok(());
            }

            // Once we have decided not to insert at the head, we can enter a
            // more sane common case: point at a linked list node and decide
            // whether to insert after it.
            let head_captured_ref =
                current_node_capture.get_captured_tvar_ref();

            let head_ref = (*head_captured_ref).as_ref();

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
            txn.capture_tvar_ref(&curr_node.next).unwrap();
        let next_val_captured_ref = next_val_cache_key.get_captured_tvar_ref();
        let next_val_ref = (*next_val_captured_ref).as_ref();
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
            let head_capture = txn.capture_tvar(&LIST_HEAD)?;
            let list_head_ref = head_capture.get_captured_tvar_ref();
            let head_node = (*list_head_ref).as_ref().unwrap();
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
            pub static ref TVAR_INT1: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(3);
            pub static ref TVAR_INT2: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(4);

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

                    *txn.capture_tvar(&TVAR_INT1)?
                        .get_captured_tvar_mut() += 1;
                    *txn.capture_tvar(&TVAR_INT2)?
                        .get_captured_tvar_mut() += 1;
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

    #[derive(Clone, Copy)]
    pub struct SizeHeader {
        size: usize
    }

    impl SizeHeader {
        fn new(size: usize) -> SizeHeader {
            SizeHeader {
                size: size
            }
        }
    }

    impl FlexibleArrayHeader for SizeHeader {
        fn get_flexible_array_len(&self) -> usize {
            self.size
        }
    }

    mod test6_state {
        use super::*;

        const TVAR1_ARRAY_INIT: [u64; 3] = [4, 5, 6];
        const TVAR2_ARRAY_INIT: [u64; 6] = [7, 8, 9, 10, 11, 12];

        lazy_static! {
            pub static ref TVAR1: VersionedTVar<SizeHeader, u64> =
                VersionedTVar::new(SizeHeader::new(3), &TVAR1_ARRAY_INIT);
            pub static ref TVAR2: VersionedTVar<SizeHeader, u64> =
                VersionedTVar::new(SizeHeader::new(6), &TVAR2_ARRAY_INIT);
        }
    }

    #[test]
    fn test6_simple_flexible_array_use() -> Result<(), TxnErrStatus> {
        use test6_state::*;
        let join_handle = thread::spawn(|| {
            VersionedTransaction::start_txn(|txn| {
                let tvar1_capture = txn.capture_tvar(&TVAR1)?;
                let tvar1_ref = tvar1_capture.get_captured_tvar_ref();

                {
                    let mut tvar2_capture = txn.capture_tvar(&TVAR2)?;
                    let mut tvar2_mut = tvar2_capture.get_captured_tvar_mut();

                    let flex_arr1_ref =
                        tvar1_ref.get_flexible_array_slice();

                    {
                        let flex_arr2_mut =
                            tvar2_mut.get_flexible_array_slice_mut();

                        for (idx, num) in flex_arr1_ref.iter().enumerate() {
                            flex_arr2_mut[idx * 2] += num;
                        }
                    }
                }

                Ok(())
            })
        });

        join_handle.join().unwrap();

        VersionedTransaction::start_txn(|txn| {
            {
                let tvar1_capture = txn.capture_tvar(&TVAR1)?;
                let tvar1_ref = tvar1_capture.get_captured_tvar_ref();

                for (idx, num) in
                    tvar1_ref
                        .get_flexible_array_slice()
                        .iter()
                        .enumerate()
                {
                    let idx_plus_four: u64 = (idx + 4).try_into().unwrap();
                    assert_eq!(idx_plus_four, *num);
                }
            }

            let tvar2_capture = txn.capture_tvar(&TVAR2)?;
            let tvar2_ref = tvar2_capture.get_captured_tvar_ref();
            let tvar2_trailing_slice =
                tvar2_ref.get_flexible_array_slice();

            assert_eq!(tvar2_trailing_slice[0], 11);
            assert_eq!(tvar2_trailing_slice[1], 8);
            assert_eq!(tvar2_trailing_slice[2], 14);
            assert_eq!(tvar2_trailing_slice[3], 10);
            assert_eq!(tvar2_trailing_slice[4], 17);
            assert_eq!(tvar2_trailing_slice[5], 12);

            Ok(())
        });

        Ok(())
    }

    mod test7_state {
        use super::*;
        use std::mem::size_of_val;

        // Note: this is a very silly class. It uses insertion sort to insert a
        // value into a growable thread-shared vector. For production, this
        // would be bad design on every level: a bad sort in insertion sort, and
        // a bad structure for insert-in-the-middle values in vector. However,
        // this is good for checking that we can grow an array and have that
        // work correctly and that we can handle high-contention situations.
        #[derive(Clone, Copy)]
        pub struct SortedVecHeader {
            size: usize,
            capacity: usize
        }

        impl SortedVecHeader {
            fn new(size: usize) -> SortedVecHeader {
                if size == 0 {
                    return SortedVecHeader {
                        size: 0,
                        capacity: 0
                    };
                }
                let num_bits_in_size = size_of_val(&size) * 8;
                let log2_size =
                    num_bits_in_size - (size - 1).leading_zeros() as usize;

                // If this is the case, the size is 0 and the capacity should
                // also be zero.

                let pow2_size_round_up = 1 << log2_size;
                assert!(pow2_size_round_up >= size);
                SortedVecHeader {
                    size,
                    capacity: pow2_size_round_up
                }
            }

            // The header after insertion
            fn get_post_insert_header(&self) -> SortedVecHeader {
                SortedVecHeader::new(self.size + 1)
            }
        }

        impl FlexibleArrayHeader for SortedVecHeader {
            fn get_flexible_array_len(&self) -> usize {
                self.capacity
            }
        }

        pub struct SortedVec<T: Copy + Ord>
            (SharedTVarRef<SortedVecHeader, T>);

        impl<T: Copy + Ord + Default + 'static> SortedVec<T> {
            pub fn new() -> SortedVec<T> {
                let empty_init = [ ];
                SortedVec {
                    0: VersionedTVar::new_shared
                        (SortedVecHeader::new(0), &empty_init)
                }
            }

            pub fn insert(&self, new_val: T) {
                VersionedTransaction::start_txn(|txn| {
                    let mut captured_inner_vec = txn.capture_tvar_ref(&self.0)?;
                    // Get the index at which we need to insert the new item. If
                    // an equivalent item is already in the list, return.
                    let insert_idx = {
                        let captured_vec_ref =
                            captured_inner_vec.get_captured_tvar_ref();
                        let capacity_slice_ref =
                            captured_vec_ref.get_flexible_array_slice();
                        let vec_size =
                            captured_vec_ref
                                .get_flexible_array_header_ref().size;
                        let size_slice_ref = &capacity_slice_ref[0..vec_size];
                        let bsearch_result =
                            size_slice_ref.binary_search(&new_val);
                        match bsearch_result {
                            Ok(_) => { return Ok(())},
                            Err(idx) => { idx }
                        }
                    };
                    let post_insert_header = {
                        captured_inner_vec
                                .get_captured_tvar_ref()
                                .get_flexible_array_header_ref()
                                .get_post_insert_header()
                    };
                    let mut possibly_resized_vec =
                        captured_inner_vec.resize_copy_prefix
                            (post_insert_header, Some(T::default()));
                    let mutable_vec_slice =
                        possibly_resized_vec.get_flexible_array_slice_mut();
                    let mut item_to_insert = new_val.clone();
                    for current_idx in insert_idx..mutable_vec_slice.len() {
                        let old_contents =
                            mutable_vec_slice[current_idx].clone();
                        mutable_vec_slice[current_idx] = item_to_insert;
                        item_to_insert = old_contents;
                    }
                    Ok(())
                })
            }

            pub fn get_regular_vec(&self) -> Vec<T> {
                VersionedTransaction::start_txn(|txn|{
                    let captured_inner_vec = txn.capture_tvar_ref(&self.0)?;
                    let captured_vec_ref =
                        captured_inner_vec.get_captured_tvar_ref();
                    let mut new_vec = Vec::<T>::new();
                    new_vec.extend_from_slice
                        (captured_vec_ref.get_flexible_array_slice());
                    new_vec.resize
                        (captured_vec_ref.get_flexible_array_header_ref().size,
                         T::default());
                    Ok(new_vec)
                })
            }
        }

        lazy_static! {
            pub static ref TVAR_VEC: SortedVec<u64> = SortedVec::new();
        }
    }

    #[test]
    fn test7_sorted_vec_insert() -> Result<(), TxnErrStatus> {
        use test7_state::*;
        use threadpool::Builder;
        use std::collections::BTreeSet;
        use rand::thread_rng;
        use rand::Rng;

        let pool = Builder::new().build();
        let mut rng = thread_rng();

        let mut oracle_set = BTreeSet::<u64>::new();

        for _count in 0..100 {
            let random_num: u64 = rng.gen::<u8>() as u64;
            oracle_set.insert(random_num);
            pool.execute(move || {
                TVAR_VEC.insert(random_num)
            })
        }

        // Wait until all jobs in the pool have finished.
        pool.join();
        assert_eq!(pool.panic_count(), 0);

        let result_vec = TVAR_VEC.get_regular_vec();

        assert_eq!(oracle_set.len(), result_vec.len());

        for (expected_item, found_item) in
            oracle_set.iter().zip(result_vec.iter()) {
            assert_eq!(expected_item, found_item);
        }

        Ok(())
    }

    mod test8_state {
        use super::*;

        lazy_static! {
            pub static ref TVAR: FixedSizeTVar<u64> =
                FixedSizeTVar::new_fixed_size(5);
        }
    }

    #[test]
    #[should_panic]
    fn test8_simultaneous_capture() {
        use test8_state::*;
        VersionedTransaction::start_txn(|txn| {
            let _capture1 = txn.capture_tvar(&TVAR)?;
            let _capture2 = txn.capture_tvar(&TVAR)?;
            Ok(())
        });
    }

    mod test9_state {
        use super::*;

        pub static TVAR: VersionedTVar<SizeHeader, u64> =
            VersionedTVar::new_empty();
    }

    #[test]
    fn test9_lazy_init_empty_tvar() {
        use test9_state::*;
        use threadpool::Builder;
        let pool = Builder::new().build();
        for _ in 0..10 {
            pool.execute(|| {
                VersionedTransaction::start_txn(|txn| {
                    let mut tvar_capture = txn.capture_tvar(&TVAR)?;
                    let trailing_array_init =
                        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36];
                    tvar_capture.fill_if_empty
                        (SizeHeader { size: 10 }, &trailing_array_init);
                    for array_item in
                        tvar_capture
                            .get_captured_tvar_mut()
                            .get_flexible_array_slice_mut()
                            .iter_mut()
                    {
                        *array_item += 1;
                    }
                    Ok(())
                });
            });
        }
        pool.join();
        VersionedTransaction::start_txn(|txn| {
            let tvar_capture = txn.capture_tvar(&TVAR)?;
            for (seen, expected) in
                tvar_capture
                    .get_captured_tvar_ref()
                    .get_flexible_array_slice()
                    .iter()
                    .zip(vec![10, 14, 18, 22, 26, 30, 34, 38, 42, 46])
            {
                assert_eq!(*seen, expected);
            }
            Ok(())
        });
    }

    mod test10_state {
        use super::*;

        pub static TVAR: VersionedTVar<SizeHeader, u64> =
            VersionedTVar::new_empty();
    }

    #[test]
    #[should_panic]
    fn test10_assert_init_twice() {
        use test10_state::TVAR;
        VersionedTransaction::start_txn(|txn| {
            let mut tvar_capture = txn.capture_tvar(&TVAR)?;
            let trailing_array_init = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36];
            tvar_capture.assert_empty_and_fill
                (SizeHeader { size: 10 }, &trailing_array_init);
             tvar_capture.assert_empty_and_fill
                (SizeHeader { size: 10 }, &trailing_array_init);
            Ok(())
        });
    }

    #[test]
    fn test11_lazy_init_empty_shared_tvar() {
        let shared_tvar_ref =
            VersionedTVar::<SizeHeader, u64>::new_shared_empty();
        use threadpool::Builder;
        let pool = Builder::new().build();
        for _ in 0..10 {
            let per_thread_tvar_ref = shared_tvar_ref.clone();
            pool.execute(move || {
                VersionedTransaction::start_txn(|txn| {
                    let mut tvar_capture =
                        txn.capture_tvar_ref(&per_thread_tvar_ref)?;
                    let trailing_array_init =
                        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36];
                    tvar_capture.fill_if_empty
                        (SizeHeader { size: 10 }, &trailing_array_init);
                    for array_item in
                        tvar_capture
                            .get_captured_tvar_mut()
                            .get_flexible_array_slice_mut()
                            .iter_mut()
                    {
                        *array_item += 1;
                    }
                    Ok(())
                });
            });
        }
        pool.join();
        VersionedTransaction::start_txn(|txn| {
            let tvar_capture = txn.capture_tvar_ref(&shared_tvar_ref)?;
            for (seen, expected) in
                tvar_capture
                    .get_captured_tvar_ref()
                    .get_flexible_array_slice()
                    .iter()
                    .zip(vec![10, 14, 18, 22, 26, 30, 34, 38, 42, 46])
            {
                assert_eq!(*seen, expected);
            }
            Ok(())
        });
    }
}

