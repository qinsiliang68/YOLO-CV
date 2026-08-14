# Readiness fix 04: unique shared claim-registry root

- Finding: a byte-identical registry descriptor could be copied to a second directory, allowing the same execution token to create a second exclusive claim.
- Red: `RED_BEHAVIOR.junit.xml` records the pre-fix failure after the registry-root binding test was added.
- Green: `GREEN_BEHAVIOR.junit.xml` records the post-fix formal execution suite.
- Fix: the active descriptor now contains `registry_root_digest`; claim occurs only when it equals the digest of the actual canonical shared root.
- Formal training, assignments, engineering gates, releases and seeds created: false.
