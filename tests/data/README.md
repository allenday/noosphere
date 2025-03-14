# Test Data

This directory contains fixtures for both unit and integration tests.

## Directory Structure

```
data/
├── private_channel/
│   └── 12345678/
│       ├── attachments/
│       │   └── photos/
│       │       └── photo_2@29-12-2017_04-47-57.jpg
│       └── messages.jsonl
└── window/
    └── window.jsonl
```

## Usage

The data in this directory is used by both unit and integration tests. The structure mimics the typical Telegram export format, with additional files for specific test cases.

- `private_channel/`: Telegram private channel exports
- `window/`: Sample Window objects in JSONL format for testing aggregation functions