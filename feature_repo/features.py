from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
import os

# absolute path to the parquet we created during preprocessing
train_path = os.path.abspath("data/processed/train.parquet")
val_path = os.path.abspath("data/processed/val.parquet")

# Use the train parquet as the source. Feast FileSource works with parquet files.
iris_file_source = FileSource(
    path=train_path,
    event_timestamp_column=None,
    created_timestamp_column=None,
)

# Define entity
sample_entity = Entity(name="sample_id", value_type=Int64, description="sample id")

# Feature view
iris_fv = FeatureView(
    name="iris_features",
    entities=["sample_id"],
    ttl=None,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=iris_file_source,
)
