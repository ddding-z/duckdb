// duckdb logical part
#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"

using namespace duckdb;

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

// retree rules part
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

#include <iomanip>
#include <regex>

//===--------------------------------------------------------------------===//
// ** ReTree Optimize Rules
//===--------------------------------------------------------------------===//

//** rule0: 分类树转回归树
class DTConvertExtension : public OptimizerExtension {
public:
	DTConvertExtension() {
		optimize_function = convertDTRule;
	}

	static bool visitExpression(Expression &expr) {
		if (expr.expression_class == ExpressionClass::BOUND_COMPARISON) {
			auto &comparison_expr = dynamic_cast<BoundComparisonExpression &>(expr);
			if (comparison_expr.left->expression_class == ExpressionClass::BOUND_FUNCTION) {
				auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
				if (func_expr.function.name == "predict") {
					auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
					std::string original_model_path = first_param.value.ToString();
					std::string opted_model_path = optimize_on_decision_tree_predicate_convert(original_model_path);
					duckdb::Value model_path_value(opted_model_path);
					first_param.value = model_path_value;
					return true;
				}
			}
		}

		bool match = false;
		ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
			if (visitExpression(child)) {
				match = true;
			}
		});
		return match;
	}

	static bool visitOperator(LogicalOperator &op) {
		for (auto &expr : op.expressions) {
			if (visitExpression(*expr)) {
				return true;
			}
		}
		for (auto &child : op.children) {
			if (visitOperator(*child)) {
				return true;
			}
		}
		return false;
	}

	static void convertDTRule(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan) {
		visitOperator(*plan);
	}
};

//===--------------------------------------------------------------------===//
// ** Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void retree_convert_rule_extension_init(duckdb::DatabaseInstance &db) {
	Connection con(db);
	auto &config = DBConfig::GetConfig(db);

	// add a parser extension: 分类树转回归树
	config.optimizer_extensions.push_back(DTConvertExtension());
	config.AddExtensionOption("clf2reg", "convert clf model to reg model", LogicalType::INVALID);

}

DUCKDB_EXTENSION_API const char *retree_convert_rule_extension_version() {
	return DuckDB::LibraryVersion();
}
}
