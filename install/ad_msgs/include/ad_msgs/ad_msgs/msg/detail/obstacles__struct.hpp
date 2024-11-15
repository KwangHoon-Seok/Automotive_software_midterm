// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/Obstacles.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'obstacles'
#include "ad_msgs/msg/detail/vehicle_output__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__Obstacles __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__Obstacles __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Obstacles_
{
  using Type = Obstacles_<ContainerAllocator>;

  explicit Obstacles_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit Obstacles_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _obstacles_type =
    std::vector<ad_msgs::msg::VehicleOutput_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::VehicleOutput_<ContainerAllocator>>>;
  _obstacles_type obstacles;

  // setters for named parameter idiom
  Type & set__obstacles(
    const std::vector<ad_msgs::msg::VehicleOutput_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::VehicleOutput_<ContainerAllocator>>> & _arg)
  {
    this->obstacles = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::Obstacles_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::Obstacles_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::Obstacles_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::Obstacles_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__Obstacles
    std::shared_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__Obstacles
    std::shared_ptr<ad_msgs::msg::Obstacles_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Obstacles_ & other) const
  {
    if (this->obstacles != other.obstacles) {
      return false;
    }
    return true;
  }
  bool operator!=(const Obstacles_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Obstacles_

// alias to use template instance with default allocator
using Obstacles =
  ad_msgs::msg::Obstacles_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__OBSTACLES__STRUCT_HPP_
