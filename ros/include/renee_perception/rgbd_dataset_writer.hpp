#ifndef RENEE_PERCEPTION__RGBD_DATASET_WRITER_HPP_
#define RENEE_PERCEPTION__RGBD_DATASET_WRITER_HPP_

#include "renee_perception/rgbd_capture_types.hpp"

namespace renee_perception
{

class RgbdDatasetWriter
{
public:
  RgbdDatasetWriter() = default;

  RgbdCaptureRecord writeCapture(const RgbdCaptureData & data) const;
};

}  // namespace renee_perception

#endif  // RENEE_PERCEPTION__RGBD_DATASET_WRITER_HPP_
