# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, private
evaluation inputs, endpoint responses, or arbitrary code execution.

Email `enes@altaysec.com.tr` with:

- affected version and command;
- a minimal reproduction using synthetic data;
- expected and observed behavior;
- potential impact;
- any suggested mitigation.

Do not include live customer data or reusable credentials. Reports will be acknowledged
as soon as practical and coordinated before public disclosure.

## Supported versions

Until the first stable release, only the latest tagged version receives security fixes.

## Threat model

The tool executes user-selected Python predictor modules and sends dataset text to
user-configured HTTP endpoints. Both are trusted operator actions. Run third-party code
and endpoints only in environments appropriate for the evaluated data.
