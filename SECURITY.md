# Security policy

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub: **Security → Report a vulnerability** on this repository. Include the version, a description, and steps to reproduce if you have them.

You'll get an acknowledgement within a few days. Fixes are released as a patch version, and the advisory is published once a fix is available.

## Supported versions

Until 1.0, only the latest minor release gets security fixes.

## Areas worth knowing about

django-llm-kit sits between your users, your database and model providers. Reports in these areas are especially useful:

- a client reaching another user's or tenant's thread, or choosing a model the touchpoint policy forbids;
- an agent tool reading or writing rows outside the caller's scope, or skipping an approval;
- provider keys stored in the database being readable after saving;
- prompt or response content being recorded when capture is off.
