# Changelog

## 1.0.0

Spend authorization signatures use tcloud. `createLocalSpendSigner` keeps the ephemeral key in an opaque handle.

Removed exports: `buildSpendAuthTypedData`, `SPEND_AUTHORIZATION_TYPE`, `SPEND_AUTH_DOMAIN_NAME`, `SPEND_AUTH_DOMAIN_VERSION`, and `wagmiSpendSigner`. Use tcloud's shielded exports for typed-data operations.
