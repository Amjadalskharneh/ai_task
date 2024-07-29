import cv2
import numpy as np

class Points:
    def __init__(self):
        self.source_points = []
        self.destination_points = []

    def add_source_point(self, x, y):
        self.source_points.append((x, y))

    def add_destination_point(self, x, y):
        self.destination_points.append((x, y))

    def get_source_points(self):
        return np.array(self.source_points, dtype=np.float32)

    def get_destination_points(self):
        return np.array(self.destination_points, dtype=np.float32)

    def clear_points(self):
        self.source_points.clear()
        self.destination_points.clear()

    def set_default_destination_points(self, scale=4):
        # Set the destination points to create a rectangle
        # The points should form a rectangle in this order: top-left, top-right, bottom-left, bottom-right
        self.destination_points = [
            (0, 0),
            (scale * 100, 0),
            (0, scale * 100),
            (scale * 100, scale * 100)
        ]

class ImageResizer:
    @staticmethod
    def resize_image(image, max_width=1280, max_height=720):
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

class PointSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Error: Unable to read the image at {image_path}")
        self.image = ImageResizer.resize_image(self.original_image)
        self.drawing = self.image.copy()
        self.points = Points()
        self.scale_x = self.original_image.shape[1] / self.image.shape[1]
        self.scale_y = self.original_image.shape[0] / self.image.shape[0]

    def create_instruction_window(self):
        instruction_image = np.ones((600, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(instruction_image, (100, 100), (300, 500), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)
        cv2.putText(instruction_image, "1", (80, 90), font, font_scale, font_color, font_thickness)
        cv2.putText(instruction_image, "2", (330, 90), font, font_scale, font_color, font_thickness)
        cv2.putText(instruction_image, "3", (80, 510), font, font_scale, font_color, font_thickness)
        cv2.putText(instruction_image, "4", (330, 510), font, font_scale, font_color, font_thickness)
        cv2.putText(instruction_image, "Please select 4 points in this order", (50, 30), font, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_image, "Press any key to start selection", (50, 550), font, 0.7, (0, 0, 0), 1)
        return instruction_image

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x, original_y = int(x * self.scale_x), int(y * self.scale_y)
            self.points.add_source_point(original_x, original_y)
            cv2.circle(self.drawing, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(self.drawing, str(len(self.points.source_points)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Select Points", self.drawing)

    def select_points(self, num_points=4):
        self.points.clear_points()
        instruction_image = self.create_instruction_window()
        cv2.imshow("Instructions", instruction_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", self.mouse_callback)
        cv2.imshow("Select Points", self.drawing)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if len(self.points.source_points) == num_points or key == ord('q'):
                break

        cv2.destroyAllWindows()

        print("Selected points (in original image coordinates):")
        for i, point in enumerate(self.points.source_points, 1):
            print(f"Point {i}: {point}")

        self.points.set_default_destination_points()
        return self.points

class MatrixCalculator:
    @staticmethod
    def calculate_transform_matrix(points):
        # Calculate and return the perspective transform matrix
        # Use cv2.getPerspectiveTransform() function
        src = points.get_source_points()
        dst = points.get_destination_points()
        return cv2.getPerspectiveTransform(src, dst)

class ImageTransformer:
    @staticmethod
    def transform_image(image, matrix, output_size=(500, 500)):
        # Apply the perspective transform to the image
        # Use cv2.warpPerspective() function
        return cv2.warpPerspective(image, matrix, output_size)

def main():
    image_path = "STOP_sign.jpg"  # Replace with your image path

    try:
        point_selector = PointSelector(image_path)
        points = point_selector.select_points()

        if len(points.source_points) == 4:
            matrix = MatrixCalculator.calculate_transform_matrix(points)
            transformed_image = ImageTransformer.transform_image(point_selector.original_image, matrix)

            cv2.imshow("Original Image", ImageResizer.resize_image(point_selector.original_image))
            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: 4 points were not selected. Unable to perform transformation.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
