import albumentations as A
import cv2

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ]
)

image = cv2.imread("./data/elon.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread("./data/mask.jpeg")

transformed = transform(image=image, mask=mask)
transformed_image = transformed['image']
transformed_mask = transformed['mask']

cv2.imshow("transformed",transformed_image)
cv2.waitKey(0)
cv2.imshow("transformed",transformed_mask)
cv2.waitKey(0)


