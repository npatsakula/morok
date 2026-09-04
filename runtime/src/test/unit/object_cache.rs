//! Object-cache store semantics, and its degradation policy: every store-side
//! failure must report a miss rather than fail the caller's compile.

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use crate::Error;
use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

fn identity() -> CompilerIdentity {
    CompilerIdentity {
        schema: OBJECT_CACHE_SCHEMA,
        backend: "fake".into(),
        target_architecture: "target-a".into(),
        toolchain: "fake 1.2.3 sha256:abc".into(),
        flags: vec!["-first".into(), "-second".into()],
        abi: "abi-v1".into(),
        object_format: "fake-object-v1".into(),
    }
}

/// Every identity field, and the order of the flag list, participates in the
/// key — as does the source itself.
#[test_case::test_case(|identity| identity.schema += 1; "schema")]
#[test_case::test_case(|identity| identity.backend.push('x'); "backend")]
#[test_case::test_case(|identity| identity.target_architecture.push('x'); "target architecture")]
#[test_case::test_case(|identity| identity.toolchain.push('x'); "toolchain")]
#[test_case::test_case(|identity| identity.flags.swap(0, 1); "flag order")]
#[test_case::test_case(|identity| identity.abi.push('x'); "abi")]
#[test_case::test_case(|identity| identity.object_format.push('x'); "object format")]
fn key_covers_every_identity_field(change: fn(&mut CompilerIdentity)) {
    let base = ObjectCacheKey::new(b"source", identity()).digest();
    let mut changed = identity();
    change(&mut changed);
    assert_ne!(ObjectCacheKey::new(b"source", changed).digest(), base);
    assert_ne!(ObjectCacheKey::new(b"other source", identity()).digest(), base);
}

#[test]
fn deterministic_hit_and_corruption_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 4096).unwrap();
    let key = ObjectCacheKey::new(b"source", identity());
    let calls = std::cell::Cell::new(0);
    let validate = |bytes: &[u8]| {
        if bytes.starts_with(b"OBJ") { Ok(()) } else { Err(Error::JitCompilation { reason: "bad fake object".into() }) }
    };
    let compile = |bytes: &'static [u8]| {
        || {
            calls.set(calls.get() + 1);
            Ok(bytes.to_vec())
        }
    };
    let first = cache.get_or_compile(&key, validate, compile(b"OBJ-one")).unwrap();
    let second = cache.get_or_compile(&key, validate, compile(b"OBJ-other")).unwrap();
    assert_eq!(first, b"OBJ-one");
    assert_eq!(second, first, "a hit serves the published bytes, not the new closure's");
    assert_eq!(calls.get(), 1, "a hit must not invoke the compile closure");

    fs::write(cache.entry_path(&key.digest()), b"corrupt").unwrap();
    let recovered = cache.get_or_compile(&key, validate, compile(b"OBJ-two")).unwrap();
    assert_eq!(recovered, b"OBJ-two");
    assert_eq!(calls.get(), 2, "a validation failure must recompile, not fail");
}

fn staging_files(dir: &std::path::Path) -> usize {
    fs::read_dir(dir)
        .unwrap()
        .filter(|entry| entry.as_ref().unwrap().path().extension().and_then(|ext| ext.to_str()) == Some("tmp"))
        .count()
}

/// Nothing serialises concurrent compilers of one key: each compiles and
/// publishes atomically, and whichever rename lands last leaves one valid,
/// readable entry behind.
#[test]
fn concurrent_writers_leave_one_valid_entry() {
    let dir = tempfile::tempdir().unwrap();
    let cache = Arc::new(ObjectCache::open(dir.path(), 4096).unwrap());
    let key = Arc::new(ObjectCacheKey::new(b"shared", identity()));
    let barrier = Arc::new(Barrier::new(8));
    let calls = Arc::new(AtomicU64::new(0));
    let threads = (0..8)
        .map(|_| {
            let (cache, key, barrier, calls) =
                (Arc::clone(&cache), Arc::clone(&key), Arc::clone(&barrier), Arc::clone(&calls));
            std::thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_compile(
                        &key,
                        |_| Ok(()),
                        || {
                            calls.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(30));
                            Ok(b"object".to_vec())
                        },
                    )
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), b"object");
    }
    assert!((1..=8).contains(&calls.load(Ordering::SeqCst)));
    assert_eq!(staging_files(dir.path()), 0, "publication must leave no staging files behind");
    assert_eq!(cache.get_or_compile(&key, |_| Ok(()), || panic!("must be a cache hit")).unwrap(), b"object");
}

/// Another compiler publishing the same key between this one's read and
/// publish is overwritten by a whole entry, never a torn one.
#[test]
fn publish_over_a_concurrent_publication_keeps_a_readable_entry() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 4096).unwrap();
    let key = ObjectCacheKey::new(b"raced", identity());
    let outer = cache
        .get_or_compile(
            &key,
            |_| Ok(()),
            || {
                assert_eq!(cache.get_or_compile(&key, |_| Ok(()), || Ok(b"OBJ-inner".to_vec())).unwrap(), b"OBJ-inner");
                Ok(b"OBJ-outer".to_vec())
            },
        )
        .unwrap();
    assert_eq!(outer, b"OBJ-outer", "a compiler returns what it compiled, not what raced it");
    assert_eq!(staging_files(dir.path()), 0);
    let entry = cache.get_or_compile(&key, |_| Ok(()), || panic!("must be a cache hit")).unwrap();
    assert_eq!(entry, b"OBJ-outer", "the last rename wins");
}

#[test]
fn eviction_bounds_stored_object_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 300).unwrap();
    for source in [b"one".as_slice(), b"two", b"three"] {
        let key = ObjectCacheKey::new(source, identity());
        cache.get_or_compile(&key, |_| Ok(()), || Ok(vec![source[0]; 100])).unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    let total: u64 = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            (entry.path().extension()?.to_str()? == "obj").then(|| entry.metadata().ok())?
        })
        .map(|metadata| metadata.len())
        .sum();
    assert!(total <= 300, "stored {total} bytes");
}

#[test]
fn unwritable_store_still_serves_compiled_bytes() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 4096).unwrap();
    let warm = ObjectCacheKey::new(b"warm", identity());
    assert_eq!(cache.get_or_compile(&warm, |_| Ok(()), || Ok(b"warm-object".to_vec())).unwrap(), b"warm-object");

    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();
    let writable = fs::File::create(dir.path().join(".writable-probe")).is_ok();
    let result = (!writable).then(|| {
        let cold = ObjectCacheKey::new(b"cold", identity());
        (
            cache.get_or_compile(&warm, |_| Ok(()), || panic!("warm entry must still be readable")),
            cache.get_or_compile(&cold, |_| Ok(()), || Ok(b"cold-object".to_vec())),
        )
    });
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    // Running as root defeats the mode bits; the assertion is then vacuous.
    let Some((warm_bytes, cold_bytes)) = result else { return };
    assert_eq!(warm_bytes.unwrap(), b"warm-object");
    assert_eq!(cold_bytes.unwrap(), b"cold-object", "an unpublishable entry must still be compiled and returned");
}

#[test]
fn uncreatable_cache_directory_disables_the_cache() {
    // procfs rejects directory creation for every uid.
    unsafe { std::env::set_var("SVOD_OBJECT_CACHE_DIR", "/proc/svod-object-cache-must-not-exist") };
    let cache = ObjectCache::from_env();
    unsafe { std::env::remove_var("SVOD_OBJECT_CACHE_DIR") };
    assert!(matches!(cache, Ok(None)), "an unopenable store must disable the cache, not fail: {cache:?}");
}
