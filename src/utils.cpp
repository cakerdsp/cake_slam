// 通用的小工具实现文件。
// 当前仅包含一个“安全的整型向量转换”函数，主要服务于 ROS2 参数系统返回 int64 数组的场景。
#include <vector>
#include <cstdint> // for int64_t
#include <limits>  // for std::numeric_limits
#include <stdexcept> // for std::out_of_range

std::vector<int> convertToIntVectorSafe(const std::vector<int64_t>& int64_vector) {
    std::vector<int> int_vector;
    int_vector.reserve(int64_vector.size()); // 预留空间以提高效率

    for (int64_t value : int64_vector) {
        // 显式做边界检查，避免静默截断造成参数错误。
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw std::out_of_range("Value is out of range for int");
        }
        int_vector.push_back(static_cast<int>(value));
    }

    // 所有元素均安全转换后返回结果。
    return int_vector;
}
