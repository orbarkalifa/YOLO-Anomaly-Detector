import numpy as np

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
        # This is a bit of a hack to make the track_ids extraction work
        # result.boxes.id.cpu().numpy() expects a single tensor
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


def test_speed_calculation():
    """
    Tests that speed is correctly calculated in pixels per second.
    """
    tracker = ObjectTracker()

    # Frame 1: An object appears
    boxes1 = MockBoxes(
        boxes_data=[{"xyxy": [[10, 10, 20, 20]], "id": [1], "cls": [0], "conf": [0.9]}]
    )
    result1 = MockResult(boxes1, {0: "person"})
    bboxes1, labels1, confidences1 = tracker.detect_objects(result1)
    tracked_objects = {}
    tracked_objects = tracker.track_objects(
        result=result1,
        bboxes=bboxes1,
        labels=labels1,
        confidences=confidences1,
        tracked_objects=tracked_objects,
        fps=30,
    )

    assert tracked_objects[1]["speed"] == 0.0, (
        "Speed should be 0 for the first detection."
    )

    # Frame 2: The object moves
    # It moves 3 pixels horizontally and 4 pixels vertically.
    # The displacement is sqrt(3^2 + 4^2) = 5 pixels.
    # At 30 FPS, dt = 1/30 seconds.
    # Speed should be 5 / (1/30) = 150 pixels/sec.
    boxes2 = MockBoxes(
        boxes_data=[{"xyxy": [[13, 14, 23, 24]], "id": [1], "cls": [0], "conf": [0.9]}]
    )
    result2 = MockResult(boxes2, {0: "person"})
    bboxes2, labels2, confidences2 = tracker.detect_objects(result2)
    tracked_objects = tracker.track_objects(
        result=result2,
        bboxes=bboxes2,
        labels=labels2,
        confidences=confidences2,
        tracked_objects=tracked_objects,
        fps=30,
    )

    expected_speed = 150.0
    assert abs(tracked_objects[1]["speed"] - expected_speed) < 1e-6, (
        f"Speed should be close to {expected_speed} pixels/second."
    )

    # Frame 3: Test with a different FPS
    # The object moves 6 pixels horizontally and 8 pixels vertically.
    # The displacement is sqrt(6^2 + 8^2) = 10 pixels.
    # At 10 FPS, dt = 1/10 seconds.
    # Speed should be 10 / (1/10) = 100 pixels/sec.
    boxes3 = MockBoxes(
        boxes_data=[{"xyxy": [[19, 22, 29, 32]], "id": [1], "cls": [0], "conf": [0.9]}]
    )
    result3 = MockResult(boxes3, {0: "person"})
    bboxes3, labels3, confidences3 = tracker.detect_objects(result3)
    tracked_objects = tracker.track_objects(
        result=result3,
        bboxes=bboxes3,
        labels=labels3,
        confidences=confidences3,
        tracked_objects=tracked_objects,
        fps=10,
    )

    expected_speed_2 = 100.0
    assert abs(tracked_objects[1]["speed"] - expected_speed_2) < 1e-6, (
        f"Speed should be close to {expected_speed_2} pixels/second for different FPS."
    )
