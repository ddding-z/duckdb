#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/operator/decimal_cast_operators.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/extension_callback.hpp"
#include "onnx/onnx_pb.h"
#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

using namespace duckdb;

//===--------------------------------------------------------------------===//
// ** Utils
//===--------------------------------------------------------------------===//

class ModelCache {
public:
	static const size_t capacity = 12;
	static std::unordered_map<std::string, std::shared_ptr<Ort::Session>> modelcache;
	static std::queue<std::string> model_queue;
	static std::shared_mutex map_mutex;

	static std::shared_ptr<Ort::Session> getOrCreateSession(const std::string &key, const Ort::Env &env,
	                                                        const Ort::SessionOptions &options) {
		{
			std::shared_lock<std::shared_mutex> sharedLock(map_mutex);
			auto it = modelcache.find(key);
			if (it != modelcache.end()) {
				return it->second;
			}
		}
		std::unique_lock<std::shared_mutex> lock(map_mutex);
		auto it = modelcache.find(key);
		if (it != modelcache.end()) {
			return it->second;
		}

		std::shared_ptr<Ort::Session> session;
		try {
			session = std::make_shared<Ort::Session>(env, key.c_str(), options);
		} catch (const Ort::Exception &e) {
			std::cerr << "Failed to create session: " << e.what() << std::endl;
			return nullptr;
		}

		if (modelcache.size() >= capacity) {
			const std::string &old_key = model_queue.front();
			modelcache.erase(old_key);
			model_queue.pop();
		}
		model_queue.push(key);
		modelcache[key] = session;

		return session;
	}
};
std::unordered_map<std::string, std::shared_ptr<Ort::Session>> ModelCache::modelcache;
std::queue<std::string> ModelCache::model_queue;
std::shared_mutex ModelCache::map_mutex;

struct InputInfo {
	std::string model_path;
	int64_t batch_size;
	int64_t num_features;
};

// input_shape: [batch_size, num_features]
static InputInfo getInputInfoFromArgs(DataChunk &args) {
	std::string input_model_path = args.GetValue(0, 0).ToString();
	int64_t batch_size = static_cast<int64_t>(args.size());
	int64_t num_features = static_cast<int64_t>(args.ColumnCount()) - 1;
	return InputInfo {input_model_path, batch_size, num_features};
}

//===--------------------------------------------------------------------===//
// ** Create Scalarfunction
//===--------------------------------------------------------------------===//
static void ModelInferenceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// args.Print();
	//** 1.get from args
	auto input_info = getInputInfoFromArgs(args);

	//** 2.create onnxruntime session
	static Ort::SessionOptions session_options;
	// session_options.DisableCpuMemArena();
	session_options.SetIntraOpNumThreads(1);
	// const ORTCHAR_T* profile_file_prefix = ("/volumn/duckdb/data/onnxruntime_profile_output/" +
	// input_info.model_path).c_str(); session_options.EnableProfiling(profile_file_prefix);
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInferenceFunction");

	std::shared_ptr<Ort::Session> session = ModelCache::getOrCreateSession(input_info.model_path, env, session_options);
	if (!session) {
		std::cerr << "Failed to create session" << std::endl;
		return;
	}
	static Ort::AllocatorWithDefaultOptions allocator;
	static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

	//** 3.create input tensor
	//   3.1.get input data
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		args.data[col_idx].Flatten(input_info.batch_size);
	}

	// ** create input tensor
	std::vector<Ort::Value> input_tensors;
	std::vector<std::string> input_names_copy;
	std::vector<const char *> inputs_names;
	int input_nums = session->GetInputCount();
	input_names_copy.reserve(input_nums);
	inputs_names.reserve(input_nums);

	std::vector<std::vector<std::string>> all_string_data;
	std::vector<std::vector<float>> all_float_data;
	std::vector<std::vector<int64_t>> all_int64_data;

	for (int i = 0; i < input_nums; i++) {
		auto input_name = session->GetInputNameAllocated(i, allocator);
		input_names_copy.emplace_back(input_name.get());
		inputs_names.emplace_back(input_names_copy.back().c_str());

		auto input_type_info = session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_tensor_info.GetShape();

		input_shape[0] = input_info.batch_size;
		auto input_type = input_tensor_info.GetElementType();
		// todo: support all data types and data cast
		if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			auto result_type = args.data[i + 1].GetType();
			if (result_type.id() == LogicalTypeId::DECIMAL) {
				std::vector<float> float_data(input_info.batch_size, 0);
				auto width = DecimalType::GetWidth(result_type);
				auto scale = DecimalType::GetScale(result_type);
				CastParameters parameters;
				switch (result_type.InternalType()) {
				case PhysicalType::INT16: {
					auto data_pointer = FlatVector::GetData<int16_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT32: {
					auto data_pointer = FlatVector::GetData<int32_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT64: {
					auto data_pointer = FlatVector::GetData<int64_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT128: {
					auto data_pointer = FlatVector::GetData<hugeint_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				default:
					throw InternalException("Unimplemented physical type for decimal");
				}
				input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
				    memory_info, float_data.data(), input_info.batch_size, input_shape.data(), input_shape.size()));
				all_float_data.emplace_back(std::move(float_data));
			} else if (result_type.id() == LogicalTypeId::FLOAT) {
				input_tensors.emplace_back(
				    Ort::Value::CreateTensor<float>(memory_info, FlatVector::GetData<float>(args.data[i + 1]),
				                                    input_info.batch_size, input_shape.data(), input_shape.size()));
			} else {
				throw std::runtime_error("Unsupported type cast to float: " + result_type.ToString());
			}
		} else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			auto strings = FlatVector::GetData<string_t>(args.data[i + 1]);
			std::vector<std::string> strings_data;
			strings_data.reserve(input_info.batch_size);
			size_t p_data_byte_count = 0;
			for (size_t i = 0; i < input_info.batch_size; i++) {
				strings_data.emplace_back(strings[i].GetString());
				p_data_byte_count += sizeof(strings_data.back());
			}
			input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, strings_data.data(), p_data_byte_count,
			                                                    input_shape.data(), input_shape.size(),
			                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
			all_string_data.emplace_back(std::move(strings_data));
		} else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
			auto result_type = args.data[i + 1].GetType();
			if (result_type.id() == LogicalTypeId::INTEGER) {
				auto data_pointer = FlatVector::GetData<int32_t>(args.data[i + 1]);
				std::vector<int64_t> int64_data(input_info.batch_size, 0);
				for (size_t i = 0; i < input_info.batch_size; i++) {
					int64_data[i] = static_cast<int64_t>(data_pointer[i]);
				}
				input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
				    memory_info, int64_data.data(), input_info.batch_size, input_shape.data(), input_shape.size()));
				all_int64_data.emplace_back(std::move(int64_data));
			} else if (result_type.id() == LogicalTypeId::BIGINT) {
				input_tensors.emplace_back(
				    Ort::Value::CreateTensor<int64_t>(memory_info, FlatVector::GetData<int64_t>(args.data[i + 1]),
				                                      input_info.batch_size, input_shape.data(), input_shape.size()));
			} else {
				throw std::runtime_error("Unsupported type cast to int64: " + result_type.ToString());
			}
		} else {
			throw std::runtime_error("Unsupported input tensor type");
		}
	}

	// ** 4. set output tensor (names, shape, type)
	int output_index = 0;
	auto output_name = session->GetOutputNameAllocated(output_index, allocator);
	std::vector<const char*> outputs_names = { output_name.get() };

	auto output_type_info = session->GetOutputTypeInfo(output_index);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_shape = output_tensor_info.GetShape();
	output_shape[0] = input_info.batch_size;

	Ort::Value output_tensor(nullptr);
	auto output_type = output_tensor_info.GetElementType();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<float>(result);
	if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		std::vector<float> output_data(input_info.batch_size);
		output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(),
		                                                output_shape.data(), output_shape.size());
		try {
			session->Run(Ort::RunOptions {nullptr}, inputs_names.data(), input_tensors.data(), input_tensors.size(),
			             outputs_names.data(), &output_tensor, 1);
		} catch (const Ort::Exception &e) {
			std::cerr << "Failed to run inference: " << e.what() << std::endl;
		}
		// ** set result data
		auto output_data_ptr = output_tensor.GetTensorMutableData<float>();
		std::memcpy(result_data, output_data_ptr, input_info.batch_size * sizeof(float));
	} else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
		std::vector<int64_t> output_data(input_info.batch_size);
		output_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, output_data.data(), output_data.size(),
		                                                  output_shape.data(), output_shape.size());
		try {
			session->Run(Ort::RunOptions {nullptr}, inputs_names.data(), input_tensors.data(), input_tensors.size(),
			             outputs_names.data(), &output_tensor, 1);
		} catch (const Ort::Exception &e) {
			std::cerr << "Failed to run inference: " << e.what() << std::endl;
		}
		// cast int64_t to float
		auto output_data_ptr = output_tensor.GetTensorMutableData<int64_t>();
		for (size_t i = 0; i < input_info.batch_size; i++) {
			result_data[i] = static_cast<float>(output_data_ptr[i]);
		}
	} else {
		throw std::runtime_error("Unsupported output tensor type");
	}
	result.Verify(input_info.batch_size);
}

static void ForgedModelInferenceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// args.Print();
	//** 1.get from args
	auto input_info = getInputInfoFromArgs(args);

	//** 2.create onnxruntime session
	static Ort::SessionOptions session_options;
	// session_options.DisableCpuMemArena();
	session_options.SetIntraOpNumThreads(1);
	// const ORTCHAR_T* profile_file_prefix = ("/volumn/duckdb/data/onnxruntime_profile_output/" +
	// input_info.model_path).c_str(); session_options.EnableProfiling(profile_file_prefix);
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInferenceFunction");

	std::shared_ptr<Ort::Session> session = ModelCache::getOrCreateSession(input_info.model_path, env, session_options);
	if (!session) {
		std::cerr << "Failed to create session" << std::endl;
		return;
	}
	static Ort::AllocatorWithDefaultOptions allocator;
	static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

	//** 3.create input tensor
	//   3.1.get input data
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		args.data[col_idx].Flatten(input_info.batch_size);
	}

	// ** create input tensor
	std::vector<Ort::Value> input_tensors;
	std::vector<std::string> input_names_copy;
	std::vector<const char *> inputs_names;
	int input_nums = session->GetInputCount();
	input_names_copy.reserve(input_nums);
	inputs_names.reserve(input_nums);

	std::vector<std::vector<std::string>> all_string_data;
	std::vector<std::vector<float>> all_float_data;
	std::vector<std::vector<int64_t>> all_int64_data;

	for (int i = 0; i < input_nums; i++) {
		auto input_name = session->GetInputNameAllocated(i, allocator);
		input_names_copy.emplace_back(input_name.get());
		inputs_names.emplace_back(input_names_copy.back().c_str());

		auto input_type_info = session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_tensor_info.GetShape();

		input_shape[0] = input_info.batch_size;
		auto input_type = input_tensor_info.GetElementType();
		// todo: support all data types and data cast
		if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			auto result_type = args.data[i + 1].GetType();
			if (result_type.id() == LogicalTypeId::DECIMAL) {
				std::vector<float> float_data(input_info.batch_size, 0);
				auto width = DecimalType::GetWidth(result_type);
				auto scale = DecimalType::GetScale(result_type);
				CastParameters parameters;
				switch (result_type.InternalType()) {
				case PhysicalType::INT16: {
					auto data_pointer = FlatVector::GetData<int16_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT32: {
					auto data_pointer = FlatVector::GetData<int32_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT64: {
					auto data_pointer = FlatVector::GetData<int64_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				case PhysicalType::INT128: {
					auto data_pointer = FlatVector::GetData<hugeint_t>(args.data[i + 1]);
					for (size_t i = 0; i < input_info.batch_size; i++) {
						TryCastFromDecimal::Operation(data_pointer[i], float_data[i], parameters, width, scale);
					}
					break;
				}
				default:
					throw InternalException("Unimplemented physical type for decimal");
				}
				input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
				    memory_info, float_data.data(), input_info.batch_size, input_shape.data(), input_shape.size()));
				all_float_data.emplace_back(std::move(float_data));
			} else if (result_type.id() == LogicalTypeId::FLOAT) {
				input_tensors.emplace_back(
				    Ort::Value::CreateTensor<float>(memory_info, FlatVector::GetData<float>(args.data[i + 1]),
				                                    input_info.batch_size, input_shape.data(), input_shape.size()));
			} else {
				throw std::runtime_error("Unsupported type cast to float: " + result_type.ToString());
			}
		} else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			auto strings = FlatVector::GetData<string_t>(args.data[i + 1]);
			std::vector<std::string> strings_data;
			strings_data.reserve(input_info.batch_size);
			size_t p_data_byte_count = 0;
			for (size_t i = 0; i < input_info.batch_size; i++) {
				strings_data.emplace_back(strings[i].GetString());
				p_data_byte_count += sizeof(strings_data.back());
			}
			input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, strings_data.data(), p_data_byte_count,
			                                                    input_shape.data(), input_shape.size(),
			                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
			all_string_data.emplace_back(std::move(strings_data));
		} else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
			auto result_type = args.data[i + 1].GetType();
			if (result_type.id() == LogicalTypeId::INTEGER) {
				auto data_pointer = FlatVector::GetData<int32_t>(args.data[i + 1]);
				std::vector<int64_t> int64_data(input_info.batch_size, 0);
				for (size_t i = 0; i < input_info.batch_size; i++) {
					int64_data[i] = static_cast<int64_t>(data_pointer[i]);
				}
				input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
				    memory_info, int64_data.data(), input_info.batch_size, input_shape.data(), input_shape.size()));
				all_int64_data.emplace_back(std::move(int64_data));
			} else if (result_type.id() == LogicalTypeId::BIGINT) {
				input_tensors.emplace_back(
				    Ort::Value::CreateTensor<int64_t>(memory_info, FlatVector::GetData<int64_t>(args.data[i + 1]),
				                                      input_info.batch_size, input_shape.data(), input_shape.size()));
			} else {
				throw std::runtime_error("Unsupported type cast to int64: " + result_type.ToString());
			}
		} else {
			throw std::runtime_error("Unsupported input tensor type");
		}
	}

	// ** 4. set output tensor (names, shape, type)
	int output_index = 0;
	auto output_name = session->GetOutputNameAllocated(output_index, allocator);
	std::vector<const char*> outputs_names = { output_name.get() };

	auto output_type_info = session->GetOutputTypeInfo(output_index);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_shape = output_tensor_info.GetShape();
	output_shape[0] = input_info.batch_size;

	Ort::Value output_tensor(nullptr);
	auto output_type = output_tensor_info.GetElementType();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<float>(result);
	if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		std::vector<float> output_data(input_info.batch_size);
		output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(),
		                                                output_shape.data(), output_shape.size());
		// try {
		// 	session->Run(Ort::RunOptions {nullptr}, inputs_names.data(), input_tensors.data(), input_tensors.size(),
		// 	             outputs_names.data(), &output_tensor, 1);
		// } catch (const Ort::Exception &e) {
		// 	std::cerr << "Failed to run inference: " << e.what() << std::endl;
		// }
		// ** set result data
		auto output_data_ptr = output_tensor.GetTensorMutableData<float>();
		std::memcpy(result_data, output_data_ptr, input_info.batch_size * sizeof(float));
	} else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
		std::vector<int64_t> output_data(input_info.batch_size);
		output_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, output_data.data(), output_data.size(),
		                                                  output_shape.data(), output_shape.size());
		// try {
		// 	session->Run(Ort::RunOptions {nullptr}, inputs_names.data(), input_tensors.data(), input_tensors.size(),
		// 	             outputs_names.data(), &output_tensor, 1);
		// } catch (const Ort::Exception &e) {
		// 	std::cerr << "Failed to run inference: " << e.what() << std::endl;
		// }
		auto output_data_ptr = output_tensor.GetTensorMutableData<int64_t>();
		for (size_t i = 0; i < input_info.batch_size; i++) {
			result_data[i] = static_cast<float>(output_data_ptr[i]);
		}
	} else {
		throw std::runtime_error("Unsupported output tensor type");
	}
	result.Verify(input_info.batch_size);
}

//===--------------------------------------------------------------------===//
// Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void retree_inference_function_extension_init(duckdb::DatabaseInstance &db) {
	// ** create onnx model inference function
	ScalarFunction model_inference_fun("predict", {LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::FLOAT,
	                                   ModelInferenceFunction);
	model_inference_fun.varargs = LogicalType::ANY;
	CreateScalarFunctionInfo model_inference_info(model_inference_fun);

	ScalarFunction forged_model_inference_fun("forge", {LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::FLOAT,
	                                   ForgedModelInferenceFunction);
	forged_model_inference_fun.varargs = LogicalType::ANY;
	CreateScalarFunctionInfo forged_model_inference_info(forged_model_inference_fun);

	// ** register a scalar function
	Connection con(db);
	auto &client_context = *con.context;
	auto &catalog = Catalog::GetSystemCatalog(client_context);
	con.BeginTransaction();

	catalog.CreateFunction(client_context, model_inference_info);
	catalog.CreateFunction(client_context, forged_model_inference_info);

	con.Commit();
}

DUCKDB_EXTENSION_API const char *retree_inference_function_extension_version() {
	return DuckDB::LibraryVersion();
}
}