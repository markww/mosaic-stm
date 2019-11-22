use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;
use std::ptr::NonNull;
use std::any::Any;
use std::sync::atomic::Ordering::AcqRel;
use std::ops::Deref;
use std::ops::DerefMut;
use std::boxed::Box;

static next_tvar_unique_id : AtomicU64 = AtomicU64::new(0);

struct TVarVersionInner<GuardedType : ?Sized> {
    refct: AtomicU64,
    inner_struct: GuardedType
}

// This is a reference to this reference counted type which does not adjust the
// reference count. This is desirable because we want reference counted copies
// to be held only by transaction objects, while the body of the transaction may
// freely refer to the tvar version without synchronizing all the time.
struct TVarImmVersionInnerPtr<GuardedType: ?Sized>
    (NonNull<TVarVersionInner<GuardedType>>);

impl<GuardedType: ?Sized> TVarImmVersionInnerPtr<GuardedType> {
    // In a TVarImmVersion, the actual data is immutable. The reference,
    // however, is mutable. We deal with this awkward juxtaposition by providing
    // a getter for the refct that only returns a mutable ref and a getter for
    // the version that only returns a const ref.
    fn get_refct(&mut self) -> &mut AtomicU64 {
        &mut (*unsafe { self.0.as_ptr() }).refct
    }

    fn get_guarded(&self) -> &GuardedType {
        & (*unsafe { self.0.as_ptr() }).inner_struct
    }
}

struct TVarImmVersion<GuardedType : ?Sized>
    (TVarImmVersionInnerPtr<GuardedType>);

impl<GuardedType : ?Sized> Drop for TVarImmVersion<GuardedType> {
    fn drop(&mut self) {
        let tvar_imm_version_ptr = self.0;
        let nonnull_to_tvar_imm_version_inner = tvar_imm_version_ptr.0;

        // I'm using AcqRel for all of these operations for now, I'll go
        // through and prove to myself that I can weaken them later.
        let prev_refct = tvar_imm_version_ptr.get_refct().fetch_sub(1, AcqRel);
        assert!(prev_refct > 0);
        // If the previous refct is equal to 1, it is our responsibility to drop
        // this inner version.
        if prev_refct == 1 {
            let to_be_dropped =
                unsafe {
                    Box::from_raw(nonnull_to_tvar_imm_version_inner.as_ptr())
                };
        }
    }
}

impl<GuardedType : Copy> Deref for TVarImmVersion<GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &Self::Target {
        self.0.get_guarded()
    }

}

impl<GuardedType : Copy> Clone for TVarImmVersion<GuardedType> {
    fn clone(&self) -> TVarImmVersion<GuardedType> {
        let tvar_imm_version_ptr = self.0;
        let prev_refct =
          tvar_imm_version_ptr.get_refct().fetch_add(1, AcqRel);
        assert!(prev_refct > 0);
        TVarImmVersion {
            0: tvar_imm_version_ptr
        }
    }
}

// A TVarShadowVersion represents a version of the object that is local to a
// transaction and which is still mutable. If the transaction fails and
// restarts, then this version will be deleted using Box's usual mechanism. If
// the transaction succeeds, then this will be converted to a TVarImmVersion so
// that it can be shared between threads without worrying about mutation.
struct TVarShadowVersion<GuardedType: ?Sized>
    (Box<TVarVersionInner<GuardedType>>);

impl<GuardedType: Copy> TVarShadowVersion<GuardedType> {

    fn new(guarded: GuardedType) -> TVarShadowVersion<GuardedType> {
        TVarShadowVersion {
            0: Box::new(TVarVersionInner {
                refct: AtomicU64::new(1),
                inner_struct: guarded
            })
        }
    }

    fn copy_from_imm_version
        (imm_version: TVarImmVersion<GuardedType>) ->
            TVarShadowVersion<GuardedType>
    {
        Self::new(*imm_version.0.get_guarded())
    }

    fn make_shadow_into_imm_version
        (shadow_version: TVarShadowVersion<GuardedType>) ->
        TVarImmVersion<GuardedType>
    {
        let raw_shadow_ptr = Box::into_raw(shadow_version.0);
        let nonnull_imm_version_ptr = NonNull::new(raw_shadow_ptr).unwrap();
        let tvar_imm_version_ptr = TVarImmVersionInnerPtr {
            0: nonnull_imm_version_ptr
        };
        TVarImmVersion {
            0: tvar_imm_version_ptr
        }
    }
}

pub struct VersionedTVar<GuardedType : ?Sized> {
    canon_ptr: RwLock<TVarImmVersion<GuardedType>>,
    this_tvar_unique_id: u64
}

impl<GuardedType : Copy> VersionedTVar<GuardedType> {
    fn new(inner_val: GuardedType) -> VersionedTVar<GuardedType> {
        let new_id : u64 = next_tvar_unique_id.fetch_add(1, AcqRel);
        assert!(new_id < std::u64::MAX);

        // To use the same flow of functions that a regular update to the TVar
        // uses, just create a shadow version and turn it into a TVarImmVersion.

        let init_shadow_version = TVarShadowVersion::new(inner_val);
        let init_imm_version =
            TVarShadowVersion::make_shadow_into_imm_version
                (init_shadow_version);

        VersionedTVar {
            canon_ptr: RwLock::new(init_imm_version),
            this_tvar_unique_id: new_id
        }
    }

    fn capture_current_version
        (&self, txn: & mut VersionedTransaction) -> CapturedTVar<GuardedType>
    {
        CapturedTVar {
            original_tvar: NonNull::from(self),
            captured_version: *self.canon_ptr.read().unwrap(),
            shadow_copy: Option::None
        }
    }
}

// A captured TVar contains the TVar's value at time of capture plus possibly
// the transaction-local shadow var. It is responsible for holding the Arc that
// describes the current transaction's reference to the TVarImmVersion. We use a
// CapturedTVarRef to allow access to the CapturedTVar within the transaction
// without having to atomically increment the reference every time.
pub struct CapturedTVar<GuardedType : ?Sized> {
    original_tvar: NonNull<VersionedTVar<GuardedType>>,
    captured_version: TVarImmVersion<GuardedType>,
    shadow_copy: Option<TVarShadowVersion<GuardedType>>
}

impl<GuardedType: Copy + ?Sized> Deref for CapturedTVar<GuardedType> {
    type Target = GuardedType;

    fn deref(&self) -> &GuardedType {
        match self.shadow_copy {
            Option::Some(shadow_version) =>
                &shadow_version.0.as_ref().inner_struct,
            Option::None => self.captured_version.0.get_guarded()
        }
    }
}

impl<GuardedType: Copy> DerefMut for CapturedTVar<GuardedType> {
    fn deref_mut(&mut self) -> &mut GuardedType {
        // If the shadow copy is none, we need to create the shadow copy.
        let shadow_version =
            match self.shadow_copy {
            Option::Some(shadow_version) => &mut shadow_version,
            Option::None => {
                let new_shadow_version =
                    TVarShadowVersion::copy_from_imm_version
                        (self.captured_version);
                self.shadow_copy = Some(new_shadow_version);
                &mut new_shadow_version
            }
        };
        &mut shadow_version.0.as_ref().inner_struct
    }
}

impl<T: Any + ?Sized> CapturedTVar<T> {

    // A wildly unsafe downcast. We basically have to do this, though, to get
    // our captured TVars out of the transaction collection.
    unsafe fn downcast_unchecked<NewType>(&mut self) ->
        &mut CapturedTVar<NewType>
    {
        &mut *(self as * mut CapturedTVar<T> as * mut CapturedTVar<NewType>)
    }
}

struct VersionedTransaction {
    // This is a set of pairs of VersionedTVar pointers and the captures.
    captured_tvar_cache: BTreeMap<u64, CapturedTVar<dyn Any>>
}

impl VersionedTransaction {
    fn get_capture_ref_for_tvar<TVarType: Copy>
        (&mut self, tvar: &VersionedTVar<TVarType>) ->
            &mut CapturedTVar<TVarType>
    {
        let this_tvar_unique_id = tvar.this_tvar_unique_id;
        let mut_any_ref_to_new_capture =
            match self.captured_tvar_cache.entry(this_tvar_unique_id) {
                Entry::Occupied(cached_captured_entry) => {
                    unsafe { cached_captured_entry.get_mut() }
                }
                Entry::Vacant(cached_vacant_entry) => {
                    let clone_of_canon_version =
                        tvar.canon_ptr.read().unwrap().clone();
                    let new_captured_tvar =
                        CapturedTVar {
                            original_tvar: NonNull::from(tvar),
                            captured_version: clone_of_canon_version,
                            shadow_copy: None
                    };
                    cached_vacant_entry.insert(new_captured_tvar)
                }
        };
        mut_any_ref_to_new_capture.downcast_unchecked::<TVarType>()
    }
}
