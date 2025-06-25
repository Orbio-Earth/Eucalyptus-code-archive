## Tile Selection CSV Files


### Overview
In ticket #868, we created the initial tile selection for the EMIT training and validation datasets. These tile selections were recorded in the following CSVs:

- initial_tiles.csv
- final_validation_set.csv

When model development started, a first draft of the EMIT CV model was trained using only the tiles from the North America region from the training and validation sets. This tile selection was recorded in:

- subset_north_america_tiles_n699.csv

For the new EMIT CV model, we will use a subset of 50% of the initial training and validation sets respectively, using random stratified sampling according to tile region. The method for this selection is described in issue #1126.

This new tile selection will be recorded in the following CSV files:

- initial_tiles_p50.csv
- final_validation_set_p50.csv

All EMIT tile selection CSVs are stored in the `src/data/no_methane_tiles/emit/csv_files` directory.

### Summary of tile CSV files:

- **initial_tiles.csv:** 100% of the tiles from the initial training tile selection done in ticket #868.
- **final_validation_set.csv:** 100% of the tiles from the final validation set tile selection done in ticket #868.
- **subset_north_america_tiles_n699.csv:** Tiles from the initial training and validation tile selection done in ticket #868, but only for the North America region.
- **initial_tiles_p50.csv:** 50% of the tiles from the initial training tile selection done in ticket #868, sampled using random stratified sampling according to tile region. Summary of method is in issue #1126.
- **final_validation_set_p50.csv:** 50% of the tiles from the final validation set tile selection done in ticket #868, sampled using random stratified sampling according to tile region. Summary of method is in issue #1126.

