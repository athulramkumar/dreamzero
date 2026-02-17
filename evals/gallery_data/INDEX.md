# DreamZero Gallery Data - Index

## Quick Stats

- **Total Tasks**: 267
- **Unique Task Descriptions**: 223
- **Galleries**: 3
- **Data Formats**: JSON, CSV, Markdown

## Files in This Directory

### 📊 Data Files

1. **evals_gallery_tasks.json** (50 KB)
   - 115 tasks from the Eval Rollout Gallery
   - Full task metadata including descriptions, categories, scenes, objects, and video URLs

2. **training_data_gallery_tasks.json** (43 KB)
   - 97 tasks from the Training Data Gallery (AgiBot)
   - Same structure as eval gallery tasks

3. **yam_gallery_tasks.json** (29 KB)
   - 55 tasks from the 30 Min Play Data Gallery (YAM)
   - Different task types focused on manipulation skills

4. **all_tasks.csv** (67 KB)
   - All 267 tasks combined in CSV format
   - Easy to import into spreadsheets, pandas, or databases
   - Columns: gallery, task_description, category, scene_name, object_name, task_name, video_url, src, poster

### 📖 Documentation Files

5. **README.md** (4.2 KB)
   - Overview of the data
   - File descriptions
   - Usage examples (Python, pandas, command line)
   - Statistics breakdown by gallery

6. **GALLERY_TASKS_SUMMARY.md** (98 KB)
   - Complete listing of all 267 tasks
   - Organized by gallery
   - Includes full details for each task
   - Distribution statistics

7. **TASK_ANALYSIS.md** (8.7 KB)
   - Detailed analysis of task patterns
   - Task type, scene type, and object type distributions
   - Gallery comparisons
   - Most common task patterns
   - Category analysis

8. **INDEX.md** (this file)
   - Quick reference guide
   - File descriptions
   - Quick access links

## Quick Access by Gallery

### Eval Rollout Gallery
- **URL**: https://dreamzero0.github.io/evals_gallery/
- **Tasks**: 115
- **Data File**: `evals_gallery_tasks.json`
- **Top Scenes**: Office (69), Kitchen (15), Lobby (15)
- **Top Tasks**: Place (28), Other (21), Pick (16)

### Training Data Gallery (AgiBot)
- **URL**: https://dreamzero0.github.io/training_data_gallery/
- **Tasks**: 97
- **Data File**: `training_data_gallery_tasks.json`
- **Top Scenes**: Kitchen (32), Home (31), Store (14)
- **Top Tasks**: Place (38), Other (15), Arrange (7)

### 30 Min Play Data Gallery (YAM)
- **URL**: https://dreamzero0.github.io/yam_gallery/
- **Tasks**: 55
- **Data File**: `yam_gallery_tasks.json`
- **Top Scenes**: Office (15), Kitchen (15), Other (15)
- **Top Tasks**: Pipe insertion (5), Cube to pad (5), Towel fold (5)

## Task Type Categories

### Manipulation Tasks (68 total)
- **Place**: 68 tasks - Placing objects in specific locations
- **Pick**: 20 tasks - Picking up objects
- **Stack**: 7 tasks - Stacking objects on top of each other
- **Pour**: 10 tasks - Pouring liquids or materials

### Cleaning Tasks (20 total)
- **Wipe**: 14 tasks - Wiping surfaces or objects
- **Clean**: 6 tasks - General cleaning tasks

### Fabric Manipulation (15 total)
- **Towel fold**: 5 tasks - Folding towels
- **T-shirt fold**: 5 tasks - Folding t-shirts
- **Pull**: 5 tasks - Pulling fabric or objects

### Assembly/Insertion Tasks (10 total)
- **Pipe insertion**: 5 tasks - Inserting pipes into slots
- **Bar to rack**: 5 tasks - Placing bars on racks

### Handoff Tasks (5 total)
- **Hand off cube**: 5 tasks - Transferring objects between hands

### Other Tasks (149 total)
- **Other**: 36 tasks - Miscellaneous tasks
- **Unknown**: 13 tasks - Unclassified tasks
- Various specialized tasks

## Scene Types

1. **Office** (95 tasks) - Desk work, office equipment
2. **Kitchen** (62 tasks) - Food preparation, dishwashing
3. **Home** (36 tasks) - General household tasks
4. **Other** (16 tasks) - Miscellaneous locations
5. **Lobby** (15 tasks) - Public spaces
6. **Store** (14 tasks) - Retail environments
7. **Bathroom** (10 tasks) - Bathroom cleaning and organization
8. **Unknown** (13 tasks) - Unclassified scenes

## Object Types

### Most Common Objects
1. **Other** (59 tasks)
2. **Kitchenware** (43 tasks)
3. **Food** (39 tasks)
4. **Cube** (16 tasks)
5. **Cleaning Tool** (10 tasks)
6. **Clothes** (10 tasks)
7. **Towel** (7 tasks)

## Usage Examples

### Load All Tasks in Python
```python
import json

with open('evals_gallery_tasks.json') as f:
    evals = json.load(f)

print(f"First task: {evals[0]['task_description']}")
print(f"Video: https://dreamzero0.github.io/evals_gallery/{evals[0]['src']}")
```

### Load CSV with Pandas
```python
import pandas as pd

df = pd.read_csv('all_tasks.csv')
print(df.groupby('gallery').size())
```

### Query Tasks by Type
```bash
# Find all "Place" tasks
grep ",Place," all_tasks.csv | wc -l

# Get all Kitchen tasks
grep ",Kitchen," all_tasks.csv > kitchen_tasks.csv
```

## Data Schema

Each task object contains:
- `task_description` - Human-readable task prompt
- `category` - Object/Task type combination
- `scene_name` - Scene type (Office, Kitchen, etc.)
- `object_name` - Object category
- `task_name` - Task type
- `src` - Relative path to video
- `poster` - Relative path to thumbnail
- `scene_id`, `object_id`, `task_id` - Internal IDs
- `info` - Raw info string from source

## Video URLs

All videos are publicly accessible at:
- Eval: `https://dreamzero0.github.io/evals_gallery/{src}`
- Training: `https://dreamzero0.github.io/training_data_gallery/{src}`
- YAM: `https://dreamzero0.github.io/yam_gallery/{src}`

## Extraction Date

Data extracted: February 16, 2026

## Notes

- Some tasks have "Unknown" scene names due to non-standard scene IDs
- The YAM gallery uses a different task taxonomy focused on specific manipulation skills
- Task descriptions may contain duplicates across different galleries
- All data was extracted from the JavaScript-embedded JSON in the gallery pages
