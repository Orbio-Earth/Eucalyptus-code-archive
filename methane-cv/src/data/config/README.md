# Data generation configurations

## Intro

Data generation runs are configured with [`hydra`](https://hydra.cc/docs/intro/), a library for composable configurations.
Logically distinct portions of a config schema are stored in their own YAML snippets and are composed at runtime by using CLI arguments.
This allows us to have YAML snippets for each satellite, each set of plumes to synthetically insert, and each split of our dataset (train, test, validation) and compose these together from the command line rather than having one complete, standalone config for every possible combination of these.
Using hydra, we can also override individual config values via the CLI without having to edit the configs or create a new one.
The final, composed config is saved locally (to `./outputs/`) and can be passed as an artifact (not currently implemented).

## Examples

Generate the training split of an EMIT dataset with synthetically inserted AVIRIS plumes:

```
python -m src.data.azure_run_data_generation satellite=emit plumes=aviris split=training
```

is functionally equivalent to the below (`split=training` and `plumes=aviris` are defaults, as specified in the top-level `config.yaml`)
```
python -m src.data.azure_run_data_generation satellite=emit
```

Generate the validation split of an S2 dataset with synthetically inserted recycled plumes (note that none of these are default values so all CLI flags are required)
```
python -m src.data.azure_run_data_generation satellite=s2 plumes=recycled split=validation
```

An example, the same as our first but where we override config values via the CLI to have no more than two plumes per chip and to apply a different concentration rescale value.

```
python -m src.data.azure_run_data_generation satellite=emit plumes=aviris split=training max_plume_count=2 plumes.concentration_rescale_value=2.5
```

Note that `max_plume_count` is in the top-level `config.yaml`, and `concentration_rescale_value` is in the "plumes" sub-configuration.
We use dot notation to show how we traverse to the value we are overriding -> `plumes.concentration_rescale_value`.

For more information and examples on using `hydra` refer to the official docs.

## Note on specialized configurations

We use the [specialized config](https://hydra.cc/docs/patterns/specializing_config/) pattern to dynamically handle config values that depend on more than one parameter.
For example the file enumerating tiles to be used is dependent on both satellite and the dataset split.
EMIT applies a different concentration rescale value to AVIRIS plumes, as specified in an override.
For this reason we have the config subdiractories `satellite_plumes/` and `satellite_split/`.
These are not meant to be defined directly in the CLI but rather are dynamically selected at runtime from the other parameters as described in the linked docs on this pattern.
