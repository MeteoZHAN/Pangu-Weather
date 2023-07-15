import os
import numpy as np
import onnx
import onnxruntime as ort
import time



# Use GPU or CPU
use_GPU = False

# The date and time of the initial field
date = '2023-07-09'
time = '08:00'

# Uncomment the model to be used
model_24 = onnx.load('models/pangu_weather_24.onnx')  # 24h
model_6 = onnx.load('models/pangu_weather_6.onnx')  # 6h
model_3 = onnx.load('models/pangu_weather_3.onnx')  # 3h
model_1 = onnx.load('models/pangu_weather_1.onnx')  # 1h

# The directory for forecasts
forecast_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"),
    ## replace to prevent invaild char ":"
    date + "-" + time.replace(":", "-"),
)

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 30

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested', }

# Initialize onnxruntime session for Pangu-Weather Models
if use_GPU:
    ort_session24 = ort.InferenceSession('models/pangu_weather_24.onnx', sess_options=options,
                                         providers=[('CUDAExecutionProvider', cuda_provider_options)])
    ort_session6 = ort.InferenceSession('models/pangu_weather_6.onnx', sess_options=options,
                                        providers=[('CUDAExecutionProvider', cuda_provider_options)])
else:
    ort_session24 = ort.InferenceSession('models/pangu_weather_24.onnx', sess_options=options,
                                         providers=['CPUExecutionProvider'])
    ort_session6 = ort.InferenceSession('models/pangu_weather_6.onnx', sess_options=options,
                                        providers=['CPUExecutionProvider'])

# Load the upper-air numpy arrays
input = np.load(os.path.join(forecast_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(forecast_dir, 'input_surface.npy')).astype(np.float32)

# Run the inference session generate per-6-hour forecast within a week.
print('Start generate per-6-hour forecast within a week')
input_24, input_surface_24 = input, input_surface
for i in range(28):
    if (i + 1) % 4 == 0:
        output, output_surface = ort_session24.run(None, {'input': input_24, 'input_surface': input_surface_24})
        input_24, input_surface_24 = output, output_surface
        # Save the results
        np.save(os.path.join(forecast_dir, 'output_upper' + str((i + 1) * 6)), output)
        np.save(os.path.join(forecast_dir, 'output_surface' + str((i + 1) * 6)), output_surface)
    else:
        output, output_surface = ort_session6.run(None, {'input': input, 'input_surface': input_surface})
        # Save the results
        np.save(os.path.join(forecast_dir, 'output_upper' + str((i + 1) * 6)), output)
        np.save(os.path.join(forecast_dir, 'output_surface' + str((i + 1) * 6)), output_surface)
    input, input_surface = output, output_surface

