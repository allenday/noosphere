# Synthetic Test Data

This directory contains synthetic fixtures for both unit and integration tests. The data is entirely fictional and does not contain any personally identifiable information.

## Directory Structure

```
data/
├── private_channel/
│   └── 12345678/
│       ├── attachments/
│       │   └── photos/
│       │       └── photo_2@15-06-2023_14-25-18.jpg
│       │       └── photo_8@22-06-2023_09-15-44.jpg
│       └── messages.jsonl
└── window/
    └── window.jsonl
```

## Usage

The data in this directory is used by both unit and integration tests. The structure mimics the typical Telegram export format, with additional files for specific test cases.

- `private_channel/`: Synthetic Telegram private channel exports with travel-themed content
- `window/`: Sample Window objects in JSONL format for testing aggregation functions

## Data Content

All data represents a fictional "Travel Explorer" channel sharing travel tips and photos. The timestamps and IDs are synthetic and do not correspond to real data.