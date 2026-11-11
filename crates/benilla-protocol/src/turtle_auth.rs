//! tortoise: the realmd login build, for vanilla and Turtle-derived realms alike.
//!
//! A stock 1.12.1 realmd takes the client's own build, 5875. Turtle WoW 1.18.1 and the realms
//! derived from it (Tortoise) accept only their own, 7272, and some custom strict-version realms
//! 12340. [`logon_with_pin_async`] presents 5875 first and, while realmd rejects the version, the
//! next build; any other refusal (account, password, transport, an offline realm) stands. Every
//! logon ([`crate::logon`], [`crate::logon_async`], [`crate::logon_with_pin`]) goes through it, so
//! each negotiates. A nonempty
//! `WOW_REALMD_BUILD` pins one build. The world server always gets [`crate::CLIENT_BUILD`]: a
//! Turtle client reports 1.18.1 to realmd and its 5875 engine to the world.

use anyhow::{bail, Context, Result};

/// The builds tried, in order.
const LOGIN_BUILDS: [u16; 3] = [crate::CLIENT_BUILD, 7272, 12340];
/// The grunt results that reject the client version: invalid, and invalid with a patch offer.
const VERSION_INVALID: u8 = 0x09;
const VERSION_UPDATE: u8 = 0x0a;

/// The full SRP6 logon and the realm list, at the first build realmd accepts: each attempt is
/// `crate::logon_with_pin_build_async` with the same `pin` and `stage` (see its contract), so a
/// PIN is asked again on each build's challenge.
pub async fn logon_with_pin_async(
    host: &str,
    username: &str,
    password: &str,
    locale: u32,
    pin: &mut impl AsyncFnMut(&[u8; 10]) -> Result<Vec<u8>>,
    stage: &mut dyn FnMut(crate::LoginStage),
) -> Result<crate::Logon> {
    let value = match std::env::var("WOW_REALMD_BUILD") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).context("reading WOW_REALMD_BUILD"),
    };
    let pinned = build_from_override(value.as_deref())?;
    let builds: &[u16] = match &pinned {
        Some(build) => std::slice::from_ref(build),
        None => &LOGIN_BUILDS,
    };
    let mut first_refusal = None;
    for &build in builds {
        let result = crate::logon_with_pin_build_async(
            host, username, password, build, locale, pin, stage,
        )
        .await;
        if !version_rejected(&result) {
            // A later build that fails for no realmd reason at all (a closed link, say) says less
            // than the version refusal that sent us there.
            return match (result, first_refusal) {
                (Err(error), Some(refusal))
                    if error.downcast_ref::<crate::AuthReject>().is_none() =>
                {
                    Err(refusal)
                }
                (result, _) => result,
            };
        }
        first_refusal.get_or_insert(result.err().expect("a refusal"));
    }
    Err(first_refusal.expect("at least one login build"))
}

/// `WOW_REALMD_BUILD`: unset or empty tries every build; otherwise one positive build number.
fn build_from_override(value: Option<&str>) -> Result<Option<u16>> {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let build = value
        .parse::<u16>()
        .context("WOW_REALMD_BUILD must be a positive 16-bit build number")?;
    if build == 0 {
        bail!("WOW_REALMD_BUILD must be a positive 16-bit build number");
    }
    Ok(Some(build))
}

/// Whether realmd refused the attempt for its client version, which lets the next build try.
fn version_rejected<T>(result: &Result<T>) -> bool {
    result.as_ref().err().is_some_and(|error| {
        error
            .downcast_ref::<crate::AuthReject>()
            .is_some_and(|reject| matches!(reject.code, VERSION_INVALID | VERSION_UPDATE))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_override_pins_one_positive_build_or_none() {
        assert_eq!(build_from_override(None).unwrap(), None);
        assert_eq!(build_from_override(Some("")).unwrap(), None);
        assert_eq!(build_from_override(Some("7272")).unwrap(), Some(7272));
        assert!(build_from_override(Some("0")).is_err());
        assert!(build_from_override(Some("70000")).is_err());
        assert!(build_from_override(Some("abc")).is_err());
    }

    #[test]
    fn only_a_version_refusal_tries_the_next_build() {
        let reject = |code| -> Result<()> {
            Err(anyhow::Error::new(crate::AuthReject {
                code,
                reply: crate::auth::LogonReply::Challenge,
            }))
        };
        assert!(version_rejected(&reject(VERSION_INVALID)));
        assert!(version_rejected(&reject(VERSION_UPDATE)));
        assert!(!version_rejected(&reject(0x04)), "a wrong password stands");
        assert!(!version_rejected(&Err::<(), _>(anyhow::anyhow!(
            "connection refused"
        ))));
        assert!(!version_rejected(&Ok(())));
    }
}
