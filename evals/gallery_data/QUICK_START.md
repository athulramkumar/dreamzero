# Quick Start Guide - DreamZero Gallery Data

## TL;DR

**267 robot manipulation tasks** extracted from 3 galleries, saved in JSON and CSV formats.

## Files You Need

- **`all_tasks.csv`** - All 267 tasks in one CSV file (easiest to use)
- **`evals_gallery_tasks.json`** - 115 tasks from Eval Rollout Gallery
- **`training_data_gallery_tasks.json`** - 97 tasks from Training Data Gallery
- **`yam_gallery_tasks.json`** - 55 tasks from YAM Gallery

## Quick Examples

### Load All Tasks (CSV)
```python
import pandas as pd
df = pd.read_csv('all_tasks.csv')
print(f"Total tasks: {len(df)}")
print(df.head())
```

### Load Specific Gallery (JSON)
```python
import json
with open('evals_gallery_tasks.json') as f:
    tasks = json.load(f)
    
# Print first task
task = tasks[0]
print(f"Task: {task['task_description']}")
print(f"Scene: {task['scene_name']}")
print(f"Video: https://dreamzero0.github.io/evals_gallery/{task['src']}")
```

### Filter by Scene Type
```python
import pandas as pd
df = pd.read_csv('all_tasks.csv')

# Get all kitchen tasks
kitchen_tasks = df[df['scene_name'] == 'Kitchen']
print(f"Kitchen tasks: {len(kitchen_tasks)}")
```

### Filter by Task Type
```python
import pandas as pd
df = pd.read_csv('all_tasks.csv')

# Get all "Place" tasks
place_tasks = df[df['task_name'] == 'Place']
print(f"Place tasks: {len(place_tasks)}")
```

### Get Video URLs
```python
import pandas as pd
df = pd.read_csv('all_tasks.csv')

# Get all video URLs
videos = df['video_url'].tolist()
print(f"Total videos: {len(videos)}")
print(videos[:5])  # First 5
```

### Count Tasks by Gallery
```python
import pandas as pd
df = pd.read_csv('all_tasks.csv')

counts = df.groupby('gallery').size()
print(counts)
```

### Command Line Examples

```bash
# Count total tasks
wc -l all_tasks.csv

# Get all "Pick" tasks
grep ",Pick," all_tasks.csv

# Count tasks by scene
cut -d',' -f4 all_tasks.csv | sort | uniq -c

# Get all video URLs
cut -d',' -f7 all_tasks.csv | tail -n +2
```

## Data Fields

Each task has:
- `task_description` - What the robot should do
- `category` - Object/Task combination
- `scene_name` - Where (Office, Kitchen, etc.)
- `object_name` - What object (Food, Kitchenware, etc.)
- `task_name` - Type of task (Pick, Place, Pour, etc.)
- `video_url` - Full URL to video
- `gallery` - Which gallery it's from

## Quick Stats

- **Total Tasks**: 267
- **Unique Descriptions**: 223
- **Task Types**: 29 (Place, Pick, Pour, Wipe, Stack, etc.)
- **Scene Types**: 8 (Office, Kitchen, Home, etc.)
- **Object Types**: 20+ (Food, Kitchenware, Clothes, etc.)

## Most Common Tasks

1. **Place** (68) - Placing objects
2. **Other** (36) - Miscellaneous
3. **Pick** (20) - Picking up objects
4. **Wipe** (14) - Cleaning
5. **Pour** (10) - Pouring liquids

## Most Common Scenes

1. **Office** (95) - 35.6%
2. **Kitchen** (62) - 23.2%
3. **Home** (36) - 13.5%

## Sample Tasks

```
"Untie shoelaces or gifts" - Gift / Untie - Office
"Stack the cube" - Cube / Stack - Office
"Brush sauce onto bread" - Kitchenware / Brush - Kitchen
"Pick up sponge and wipe shoe" - Cleaning Tool / Wipe - Bathroom
"Fold the t-shirt" - Shirt / T-shirt fold - Other
```

## Need More Info?

- **README.md** - Detailed documentation
- **INDEX.md** - Complete reference
- **TASK_ANALYSIS.md** - Statistical analysis
- **GALLERY_TASKS_SUMMARY.md** - All 267 tasks listed
- **EXTRACTION_REPORT.md** - Full extraction report

## Gallery URLs

- Eval: https://dreamzero0.github.io/evals_gallery/
- Training: https://dreamzero0.github.io/training_data_gallery/
- YAM: https://dreamzero0.github.io/yam_gallery/
