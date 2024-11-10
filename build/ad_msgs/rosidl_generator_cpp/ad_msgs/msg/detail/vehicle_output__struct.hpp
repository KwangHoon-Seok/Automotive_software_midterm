// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/VehicleOutput.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__VEHICLE_OUTPUT__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__VEHICLE_OUTPUT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__VehicleOutput __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__VehicleOutput __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct VehicleOutput_
{
  using Type = VehicleOutput_<ContainerAllocator>;

  explicit VehicleOutput_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = "";
      this->x = 0.0;
      this->y = 0.0;
      this->yaw = 0.0;
      this->velocity = 0.0;
      this->length = 0.0;
      this->width = 0.0;
    }
  }

  explicit VehicleOutput_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = "";
      this->x = 0.0;
      this->y = 0.0;
      this->yaw = 0.0;
      this->velocity = 0.0;
      this->length = 0.0;
      this->width = 0.0;
    }
  }

  // field types and members
  using _id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _id_type id;
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _velocity_type =
    double;
  _velocity_type velocity;
  using _length_type =
    double;
  _length_type length;
  using _width_type =
    double;
  _width_type width;

  // setters for named parameter idiom
  Type & set__id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__velocity(
    const double & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__length(
    const double & _arg)
  {
    this->length = _arg;
    return *this;
  }
  Type & set__width(
    const double & _arg)
  {
    this->width = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::VehicleOutput_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::VehicleOutput_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::VehicleOutput_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::VehicleOutput_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__VehicleOutput
    std::shared_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__VehicleOutput
    std::shared_ptr<ad_msgs::msg::VehicleOutput_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const VehicleOutput_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->length != other.length) {
      return false;
    }
    if (this->width != other.width) {
      return false;
    }
    return true;
  }
  bool operator!=(const VehicleOutput_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct VehicleOutput_

// alias to use template instance with default allocator
using VehicleOutput =
  ad_msgs::msg::VehicleOutput_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__VEHICLE_OUTPUT__STRUCT_HPP_
