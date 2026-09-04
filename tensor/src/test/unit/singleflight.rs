use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::singleflight::Singleflight;

#[test]
fn try_claim_is_exclusive_until_the_ticket_drops() {
    let flight = Singleflight::new();
    let ticket = flight.try_claim(1u32).expect("first claim wins");
    assert!(flight.try_claim(1u32).is_none());
    assert!(flight.try_claim(2u32).is_some());
    drop(ticket);
    assert!(flight.try_claim(1u32).is_some());
}

#[test]
fn run_waits_for_a_claimed_key_and_takes_the_published_value() {
    let flight = Arc::new(Singleflight::new());
    let published = Arc::new(Mutex::new(None::<u32>));
    let ticket = flight.try_claim(7u32).unwrap();

    let waiter = std::thread::spawn({
        let (flight, published) = (Arc::clone(&flight), Arc::clone(&published));
        move || {
            flight.run(
                7u32,
                || *published.lock(),
                || -> Result<u32, ()> { panic!("a value published by the winner must not be recomputed") },
            )
        }
    });
    std::thread::sleep(Duration::from_millis(50));
    assert!(!waiter.is_finished(), "the loser must park while the ticket is held");

    *published.lock() = Some(42);
    drop(ticket);
    assert_eq!(waiter.join().unwrap(), Ok(42));
}

#[test]
fn a_failed_winner_hands_the_key_to_the_next_caller() {
    let flight = Singleflight::new();
    let failed: Result<u32, &str> = flight.run(3u32, || None, || Err("compile failed"));
    assert_eq!(failed, Err("compile failed"));
    assert_eq!(flight.run(3u32, || None, || Ok::<_, &str>(5)), Ok(5));
}
