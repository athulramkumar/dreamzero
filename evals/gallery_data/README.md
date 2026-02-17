# DreamZero Gallery Tasks Data

This directory contains comprehensive data extracted from the three DreamZero task galleries.

## Overview

- **Total Tasks**: 267 (115 + 97 + 55)
- **Data Sources**:
  1. [Eval Rollout Gallery](https://dreamzero0.github.io/evals_gallery/) - 115 tasks
  2. [Training Data Gallery (AgiBot)](https://dreamzero0.github.io/training_data_gallery/) - 97 tasks
  3. [30 Min Play Data Gallery (YAM)](https://dreamzero0.github.io/yam_gallery/) - 55 tasks

## Files

### JSON Files (Detailed Data)

- `evals_gallery_tasks.json` - All 115 tasks from the Eval Rollout Gallery
- `training_data_gallery_tasks.json` - All 97 tasks from the Training Data Gallery
- `yam_gallery_tasks.json` - All 55 tasks from the YAM Gallery

Each JSON file contains an array of task objects with the following fields:
- `task_description` - Human-readable description of the task
- `category` - Object type / Task type combination
- `scene_name` - Scene type (Office, Kitchen, Lobby, Bathroom, Store, Home, Outdoor, Other)
- `object_name` - Object category (Food, Kitchenware, Clothes, etc.)
- `task_name` - Task type (Pick, Place, Pour, Wipe, Stack, etc.)
- `src` - Relative path to video file
- `poster` - Relative path to poster/thumbnail image
- `scene_id`, `object_id`, `task_id` - Internal IDs

### CSV File (All Tasks Combined)

- `all_tasks.csv` - All 267 tasks from all three galleries in CSV format

Columns:
- `gallery` - Source gallery name
- `task_description` - Task description/prompt
- `category` - Object/Task category
- `scene_name` - Scene type
- `object_name` - Object type
- `task_name` - Task type
- `video_url` - Full URL to video
- `src` - Relative video path
- `poster` - Relative poster path

### Summary Document

- `GALLERY_TASKS_SUMMARY.md` - Comprehensive markdown document with all tasks, organized by gallery, including statistics and distributions

## Task Statistics

### Eval Rollout Gallery (115 tasks)

**Top Scene Types:**
- Office: 69 tasks
- Kitchen: 15 tasks
- Lobby: 15 tasks

**Top Object Types:**
- Other: 34 tasks
- Kitchenware: 27 tasks
- Food: 17 tasks

**Top Task Types:**
- Place: 28 tasks
- Other: 21 tasks
- Pick: 16 tasks

### Training Data Gallery (97 tasks)

**Top Scene Types:**
- Kitchen: 32 tasks
- Home: 31 tasks
- Store: 14 tasks

**Top Object Types:**
- Other: 25 tasks
- Food: 22 tasks
- Kitchenware: 15 tasks

**Top Task Types:**
- Place: 38 tasks
- Other: 15 tasks
- Arrange: 7 tasks

### YAM Gallery (55 tasks)

**Top Scene Types:**
- Office: 15 tasks
- Kitchen: 15 tasks
- Other: 15 tasks

**Top Object Types:**
- Cube: 10 tasks
- Pipe: 5 tasks
- Towel: 5 tasks

**Top Task Types:**
- Pipe insertion: 5 tasks
- Cube to pad: 5 tasks
- Towel fold: 5 tasks

## Usage Examples

### Python - Load JSON Data

```python
import json

# Load eval tasks
with open('evals_gallery_tasks.json', 'r') as f:
    evals_tasks = json.load(f)

# Print first task
task = evals_tasks[0]
print(f"Task: {task['task_description']}")
print(f"Category: {task['category']}")
print(f"Scene: {task['scene_name']}")
print(f"Video: https://dreamzero0.github.io/evals_gallery/{task['src']}")
```

### Python - Load CSV Data

```python
import pandas as pd

# Load all tasks
df = pd.read_csv('all_tasks.csv')

# Filter by scene
kitchen_tasks = df[df['scene_name'] == 'Kitchen']

# Group by gallery
gallery_counts = df.groupby('gallery').size()
print(gallery_counts)
```

### Command Line - Query Tasks

```bash
# Count tasks by scene
cut -d',' -f4 all_tasks.csv | sort | uniq -c

# Find all "Pick" tasks
grep ",Pick," all_tasks.csv

# Get all video URLs
cut -d',' -f7 all_tasks.csv | tail -n +2
```

## Data Extraction Method

The data was extracted from the JavaScript-based gallery pages by:
1. Navigating to each gallery URL
2. Parsing the embedded JSON data from the page source
3. Mapping internal IDs to human-readable names
4. Constructing full video URLs
5. Organizing and exporting to JSON and CSV formats

## Notes

- All video URLs are publicly accessible
- The galleries use a consistent structure across all three sites
- Some tasks may have "Unknown" scene names if they use non-standard scene IDs
- The YAM gallery has a different set of object and task types compared to the other two galleries
