from collections import deque
from scipy.optimize import linear_sum_assignment
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class FaceTracker:
    def __init__(self, max_disappeared=5, history_size=10, iou_threshold=0.4):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.max_disappeared = max_disappeared
        self.history_size = history_size
        self.iou_threshold = iou_threshold

    def register(self, box, name):
        self.objects[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque([name], maxlen=self.history_size)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def update(self, boxes, names):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_stable_names()

        if len(self.objects) == 0:
            for i in range(len(boxes)):
                self.register(boxes[i], names[i])
        else:
            object_ids = list(self.objects.keys())
            object_boxes = list(self.objects.values())

            iou_matrix = np.zeros((len(boxes), len(object_ids)), dtype=np.float32)

            for i, box in enumerate(boxes):
                for j, object_box in enumerate(object_boxes):
                    iou_matrix[i, j] = iou(box, object_box)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            used_rows = set()
            used_cols = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    continue

                object_id = object_ids[c]
                self.objects[object_id] = boxes[r]
                self.disappeared[object_id] = 0
                self.history[object_id].append(names[r])

                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(len(boxes))) - used_rows
            unused_cols = set(range(len(object_ids))) - used_cols

            for c in unused_cols:
                object_id = object_ids[c]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for r in unused_rows:
                self.register(boxes[r], names[r])

        return self.get_stable_names()

    def get_stable_names(self):
        stable_objects = {}
        for object_id, hist in self.history.items():
            if self.disappeared.get(object_id, 0) == 0:
                # Get the most common name if it's not 'Unknown' or if it's the only option
                most_common_name = max(set(hist), key=list(hist).count)
                
                # Rule: Only confirm a name if it appears in more than half of recent frames
                # and is not 'Unknown', unless 'Unknown' is the overwhelming majority.
                if most_common_name != "Unknown" and list(hist).count(most_common_name) > self.history_size // 2:
                    final_name = most_common_name
                else:
                    # Fallback to the most recent detection if not stable
                    final_name = hist[-1] if hist else "Unknown"

                stable_objects[object_id] = (self.objects[object_id], final_name)
        return stable_objects