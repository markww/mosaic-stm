use indexmap::IndexMap;
use indexmap::map::Entry;
use std::sync::atomic::AtomicPtr;
use std::sync::RwLock;
use std::ptr::NonNull;
use std::any::Any;
use std::any::TypeId;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::spin_loop_hint;
use std::ops::Deref;
use std::ops::DerefMut;
use std::usize;
use std::marker::PhantomData;
use std::alloc::Layout;
use std::alloc::System;
use std::alloc::GlobalAlloc;
use std::collections::HashMap;
use std::ptr::null_mut;

// The inner form of a TVarVersion is the type itself plus a pointer to "next".
// These versions will be, at various stages in their lifetime, inserted into
// linked lists, and so we provide that extra pointer-width.
#[derive(Copy)]
#[derive(Clone)]
struct TVarVersionInner<GuardedType> {
    next_ptr: Option<NonNull<GuardedType>>,
    inner_struct: GuardedType
}

#[derive(Copy)]
#[derive(Clone)]
struct TypeIdAnnotatedPtr {
    type_id: TypeId,
    ptr: * mut u8
}

static id_to_layout_map:
    RwLock<HashMap<TypeId, Layout>> = RwLock::new(HashMap::new());

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
    unsafe {
        let write_lock_guard = id_to_layout_map.write().unwrap();
        match write_lock_guard.insert(this_type_typeid, this_type_layout) {
            Some(already_present_result) => {
                // This is possible because two threads may have simultaneously
                // read the map and decided that an entry needed to be inserted.
                // Just check that the item already there is compatible with
                // what we wanted to insert.
                assert_eq!(already_present_result, this_type_layout);
            },
            None => {
                // The key was not present, nothing to do.
            }
        }
    };
    result_pair
}

fn get_layout_for_type_id_assert_present(typeid: TypeId) -> Layout {
    let read_lock_guard = id_to_layout_map.read().unwrap();
    *read_lock_guard.get(&typeid).unwrap()
}

impl TypeIdAnnotatedPtr {

    fn alloc<T: 'static>() -> TypeIdAnnotatedPtr {
        let (typeid, layout) = idempotent_add_type_get_layout_and_id::<T>();
        let new_ptr = System.alloc(layout);
        assert!(new_ptr != null_mut::<u8>());
        TypeIdAnnotatedPtr {
            type_id: typeid,
            ptr: new_ptr
        }
    }

    fn dealloc(&self) {
        // We have laid out the TVarVersionInner object such that the layout is
        // placed at the start of the object. This means that we can always grab
        // it, even without knowing for certain what the real type is.
        System.dealloc
            (self.ptr, get_layout_for_type_id_assert_present(self.type_id));
    }

    fn dynamic_cast<T: 'static + ?Sized>(self) -> Option<* mut T> {
        let target_type_id = TypeId::of::<T>();
        if target_type_id == self.type_id {
            Some(self.ptr as * mut T)
        } else {
            None
        }
    }
}

#[derive(Copy)]
#[derive(Clone)]
struct TVarImmVersion<GuardedType: ?Sized> {
    type_erased_inner: TypeIdAnnotatedPtr,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType : 'static> Deref for TVarImmVersion<GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        let downcast_inner_ptr =
            self.type_erased_inner
                .dynamic_cast::<TVarVersionInner<GuardedType>>().unwrap();
        &unsafe { *downcast_inner_ptr }.inner_struct
    }
}

// A TVarShadowVersion represents a version of the object that is local to a
// transaction and which is still mutable. If the transaction fails and
// restarts, then this version will be dropped. If the transaction succeeds,
// then this will be converted to a TVarImmVersion so that it can be shared
// between threads without worrying about mutation.
struct TVarShadowVersion<GuardedType: ?Sized> {
    type_erased_inner: TypeIdAnnotatedPtr,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType: ?Sized> Drop for TVarShadowVersion<GuardedType> {
    fn drop(&mut self) {
        self.type_erased_inner.dealloc();
    }
}

impl<GuardedType: Copy + 'static + ?Sized> TVarShadowVersion<GuardedType> {

    fn new(guarded: GuardedType) -> TVarShadowVersion<GuardedType> {
        let inner_annot_ptr = TypeIdAnnotatedPtr::alloc::<GuardedType>();
        let cast_inner_ptr =
            inner_annot_ptr.dynamic_cast::<GuardedType>().unwrap();
        unsafe { *cast_inner_ptr = guarded };
        TVarShadowVersion {
            type_erased_inner: inner_annot_ptr,
            phantom: PhantomData { }
        }
    }

    fn copy_from_imm_version
        (imm_version: TVarImmVersion<GuardedType>) ->
            TVarShadowVersion<GuardedType>
    {
        Self::new(*imm_version)
    }

    fn make_shadow_into_imm_version
        (shadow_version: TVarShadowVersion<GuardedType>) ->
        TVarImmVersion<GuardedType>
    {
        let typeid_annotated_shadow_ptr = shadow_version.type_erased_inner;
        std::mem::forget(shadow_version);
        TVarImmVersion {
            type_erased_inner: typeid_annotated_shadow_ptr,
            phantom: PhantomData { }
        }
    }
}

impl<GuardedType: 'static> Deref for TVarShadowVersion<GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        &*self.type_erased_inner.dynamic_cast::<GuardedType>().unwrap()
    }
}

impl<GuardedType: 'static> DerefMut for TVarShadowVersion<GuardedType> {
    fn deref_mut(&mut self) -> &mut GuardedType {
        &mut *self.type_erased_inner.dynamic_cast::<GuardedType>().unwrap()
    }
}

struct VersionedTVarTypeErased {
    canon_ptr: AtomicPtr<u8>,
    stale_version_list: AtomicPtr<u8>,
    type_id: TypeId
}

pub struct VersionedTVar<GuardedType : ?Sized> {
    inner_type_erased: VersionedTVarTypeErased,
    phantom: PhantomData<GuardedType>
}

impl<GuardedType : Copy + 'static> VersionedTVar<GuardedType> {
    fn new(inner_val: GuardedType) -> VersionedTVar<GuardedType> {
        // To use the same flow of functions that a regular update to the TVar
        // uses, just create a shadow version and turn it into a TVarImmVersion.

        let init_shadow_version = TVarShadowVersion::new(inner_val);
        let init_imm_version =
            TVarShadowVersion::make_shadow_into_imm_version
                (init_shadow_version);

        let init_imm_version_annot_ptr = init_imm_version.type_erased_inner;
        let result: VersionedTVar<GuardedType> =
            VersionedTVar {
                inner_type_erased: VersionedTVarTypeErased {
                    canon_ptr:
                        AtomicPtr::new
                            (init_imm_version_annot_ptr
                                .dynamic_cast::<GuardedType>()
                                .unwrap() as * mut u8),
                    stale_version_list: AtomicPtr::new(null_mut()),
                    type_id: init_imm_version_annot_ptr.type_id
                },
                phantom: PhantomData { }
            };
        result
    }

    fn get_current_canon_ptr(&self) -> NonNull<TVarVersionInner<GuardedType>> {
        // null is used to indicate that the canonical version is undergoing an
        // update but is mid-transaction. If we see this, spin until we get a
        // non-null value.
        let tvar_typeid = self.inner_type_erased.type_id;
        assert!(tvar_typeid == TypeId::of::<GuardedType>());
        let mut current_canon_opt = NonNull::new(null_mut());
        while current_canon_opt.is_none() {
            let current_canon_ptr =
                self.inner_type_erased.canon_ptr.load(Acquire);
            current_canon_opt = NonNull::new(current_canon_ptr);
            spin_loop_hint();
        }
        current_canon_opt.unwrap().cast::<TVarVersionInner<GuardedType>>()
    }
}

// A captured TVar contains the TVar's value at time of capture plus possibly
// the transaction-local shadow var if the txn has performed a write (or at
// least prepared to by getting a mutable ref).
pub struct CapturedTVar<GuardedType : ?Sized> {
    captured_version: TVarImmVersion<GuardedType>,
    shadow_copy: Option<TVarShadowVersion<GuardedType>>
}

impl<GuardedType: Copy + 'static> CapturedTVar<GuardedType> {
    fn get_shadow_copy_create_if_not_present(&mut self)
        -> &mut TVarShadowVersion<GuardedType>
    {
        match self.shadow_copy {
            Some(already_present_shadow_copy) => {
                &mut already_present_shadow_copy
            },
            None => {
                let ptr_to_tvar_version_inner =
                    self.captured_version
                        .type_erased_inner
                        .dynamic_cast::<TVarVersionInner<GuardedType>>()
                        .unwrap();
                let copy_of_inner_struct =
                    (*ptr_to_tvar_version_inner).inner_struct;
                let new_shadow_version =
                    TVarShadowVersion::new(copy_of_inner_struct);
                self.shadow_copy = Some(new_shadow_version);
                &mut new_shadow_version
            }
        }
    }
}

#[derive(Clone)]
#[derive(Copy)]
struct CapturedTVarRef<'txn, GuardedType> {
    captured_tvar_idx: usize,
    transaction: &'txn VersionedTransaction,
    phantom: PhantomData<GuardedType>
}

impl<'txn, GuardedType: 'static> Deref for CapturedTVarRef<'txn, GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        let tvar_capture =
            &self.transaction.captured_tvar_cache
                .get_index(self.captured_tvar_idx).unwrap().1;
        let inner_typeid_annot_ptr =
            match &tvar_capture.shadow_copy {
                Some(present_shadow_version) => {
                    present_shadow_version.type_erased_inner
                }
                None => {
                    tvar_capture.captured_version.type_erased_inner
                }
            };
        &(*inner_typeid_annot_ptr
            .dynamic_cast::<TVarVersionInner<GuardedType>>()
            .unwrap()).inner_struct
    }
}

impl<'txn, GuardedType: 'static + Copy> DerefMut for
    CapturedTVarRef<'txn, GuardedType>
{
    fn deref_mut(&mut self) -> &mut GuardedType {
        let tvar_capture =
            self.transaction.captured_tvar_cache
                .get_index(self.captured_tvar_idx).unwrap().1;
        let shadow_copy =
            (*tvar_capture).get_shadow_copy_create_if_not_present();
        // Left off here
    }
}

struct VersionedTransaction {
    // This associates tvars with captures so that we can look up the captured
    // values without constantly grabbing locks or using atomic operations. We
    // use IndexMap to get a stable index that we can use throught the
    // transaction. At the end, we will sort by tvar pointer order to provide
    // a canonical locking order to prevent deadlocks when updating tvars.
    captured_tvar_cache:
        IndexMap<NonNull<VersionedTVar<dyn Any>>, CapturedTVar<dyn Any>>
}

impl VersionedTransaction {
    fn get_capture_ref_for_tvar<TVarType: Copy>
        (&mut self, tvar: &VersionedTVar<TVarType>) ->
            CapturedTVarRef<TVarType>
    {
        let nonnull_ptr_to_type_erased_tvar = NonNull::from(&tvar);
        let index =
            match
                self.captured_tvar_cache.entry(nonnull_ptr_to_type_erased_tvar)
            {
                Entry::Occupied(cached_captured_entry) =>
                    cached_captured_entry.index(),
                Entry::Vacant(cached_vacant_entry) => {
                    let new_type_erased_captured_tvar =
                        tvar.capture_current_version();
                    let new_index = cached_vacant_entry.index();
                    cached_vacant_entry.insert(new_type_erased_captured_tvar);
                    new_index
                }
            };
        CapturedTVarRef {
            captured_tvar_idx: index,
            transaction: self,
            phantom: PhantomData { }
        }
    }
}
