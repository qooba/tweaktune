import pyarrow as pa
from .tweaktune import Step, StepConfig, Jsonl, Parquet, Csv, Arrow, Lang

def hello():
    return "Hello, World!"

def get_buffer(step: Step):
    b = step.create_arrow_buffer()
    reader = pa.ipc.open_stream(b)

    d = reader.read_all()
    return d.to_pandas()

def read_buffer(step: Step, data: pa.lib.Table):
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, data.schema)
    writer.write_table(data)
    writer.close()
    buffer = sink.getvalue().to_pybytes()
    step.read_pyarrow(buffer)

