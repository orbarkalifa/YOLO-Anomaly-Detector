import numpy as np

from src.config import ProjectConfig
from src.tracking.tracker import ObjectTracker


class MockResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names
        self.orig_img = np.zeros((100, 100, 3), dtype=np.uint8)


class MockBoxes:
    def __init__(self, boxes_data):
        self._boxes = []
        for data in boxes_data:
            self._boxes.append(MockBox(**data))

    def __iter__(self):
        return iter(self._boxes)

    @property
    def id(self):
        ids = [box.id.numpy()[0] for box in self._boxes]
        return MockTensor(ids)


class MockBox:
    def __init__(self, xyxy, id, cls, conf):
        self.xyxy = MockTensor(xyxy)
        self.id = MockTensor(id)
        self.cls = MockTensor(cls)
        self.conf = MockTensor(conf)


class MockTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.data)


def test_center_history_trimming():
    """
    Tests that the 'centers' list for tracked objects is trimmed
    according to max_center_history in the config.
    """
    tracker = ObjectTracker()

    # Create a mock config with a small max_center_history
    mock_config = ProjectConfig()
    mock_config.max_center_history = 5

    tracked_objects = {}
    fps = 30

    # Simulate tracking an object for more frames than max_center_history
    for i in range(10):  # 10 frames > max_center_history (5)
        boxes_data = [
            {
                "xyxy": [[10 + i, 10 + i, 20 + i, 20 + i]],
                "id": [1],
                "cls": [0],
                "conf": [0.9],
            }
        ]
        boxes = MockBoxes(boxes_data)
        result = MockResult(boxes, {0: "person"})

        bboxes, labels, confidences = tracker.detect_objects(result)
        tracked_objects = tracker.track_objects(
            result=result,
            bboxes=bboxes,
            labels=labels,
            confidences=confidences,
            tracked_objects=tracked_objects,
            fps=fps,
            config=mock_config,
        )

    # Assert that the centers list is trimmed to max_center_history
    assert len(tracked_objects[1]["centers"]) == mock_config.max_center_history, (
        f"Centers list length should be {mock_config.max_center_history}, but got {len(tracked_objects[1]['centers'])}"
    )

    # Check that the oldest entries were removed (i.e., the list contains the last N centers)
    # The centers list should contain centers from i=5 to i=9, where the center for frame `i` is (15+i, 15+i)
    expected_centers = [(15 + i, 15 + i) for i in range(5, 10)]
    assert tracked_objects[1]["centers"] == expected_centers, (
        f"Centers list content is incorrect. Expected {expected_centers}, got {tracked_objects[1]['centers']}"
    )
