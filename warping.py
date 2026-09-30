import cv2
import numpy as np

# ---------------------------------
# Configuration
# ---------------------------------

IMAGE_PATH = "ChArUco.jpg"

WIDTH = 800
HEIGHT = 500

points = []


# ---------------------------------
# Mouse Click Function
# ---------------------------------

def mouse_callback(event, x, y, flags, param):

    global image

    if event == cv2.EVENT_LBUTTONDOWN:

        # Store clicked point
        points.append([x, y])

        print(f"Point {len(points)} : ({x}, {y})")

        # Draw point
        cv2.circle(image, (x, y), 6, (0, 0, 255), -1)

        # Draw point number
        cv2.putText(
            image,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.imshow("Select 4 Corners", image)

        # If 4 points selected
        if len(points) == 4:

            src = np.float32(points)

            dst = np.float32([
                [0, 0],
                [WIDTH, 0],
                [WIDTH, HEIGHT],
                [0, HEIGHT]
            ])

            H = cv2.getPerspectiveTransform(src, dst)

            warped = cv2.warpPerspective(
                original,
                H,
                (WIDTH, HEIGHT)
            )

            print("\nSource Points")
            print(src)

            print("\nHomography Matrix")
            print(H)

            cv2.imshow("Warped Image", warped)

            print("\nPress any key to exit...")
            cv2.waitKey(0)

            cv2.destroyAllWindows()


# ---------------------------------
# Main
# ---------------------------------

original = cv2.imread(IMAGE_PATH)

if original is None:
    print("Image not found!")
    exit()

image = original.copy()

print("-------------------------------------")
print("Click the corners in this order:")
print("1. Top Left")
print("2. Top Right")
print("3. Bottom Right")
print("4. Bottom Left")
print("-------------------------------------")

cv2.namedWindow("Select 4 Corners")
cv2.setMouseCallback("Select 4 Corners", mouse_callback)

cv2.imshow("Select 4 Corners", image)

cv2.waitKey(0)

cv2.destroyAllWindows()