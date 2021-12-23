import cv2
import uuid
import os
from .constants import UPLOAD_FOLDER
from .logger import logger


class img_handler:
    def load(self, img_url):
        """load image and return image as numpy array

        Args:
            img_url ([string]): address of image in file system or database
        """
        pass

    def save(self, img):
        """Save img

        Args:
            img ([numpy.ndarray]): numpy.ndarray as image
        """
        pass

    def save_file(self, file):
        """Save file

        Args:
            file (<class 'werkzeug.datastructures.FileStorage>)
        """
        pass

    def delete(self, img_url):
        """delete image using image url

        Args:
            img_url ([string]): address of image in file system or database
        """
        pass


class img_handler_fs(img_handler):
    """Handle image stored in file system"""

    def __init__(self):
        super(img_handler_fs, self).__init__()
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)

    def load(self, img_url):
        fullpath = "/".join([UPLOAD_FOLDER, img_url])

        # OpenCV works with BGR instead of RGB
        return cv2.cvtColor(cv2.imread(fullpath), cv2.COLOR_BGR2RGB)

    def save(self, img):
        # Save img with an unique file name
        ext = ".png"
        filename = str(uuid.uuid4()) + ext
        fullpath = "/".join([UPLOAD_FOLDER, filename])

        # OpenCV works with BGR in instead of RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fullpath, img)

        # Create the image url
        return filename

    def save_file(self, file):
        """File

        Args:
            file (<class 'werkzeug.datastructures.FileStorage>)

        Returns:
            [string]: the unique filename of the stored file
        """
        filename = file.filename
        ext = os.path.splitext(filename)
        filename = str(uuid.uuid4()) + ext[-1]
        destination = "/".join([UPLOAD_FOLDER, filename])
        file.save(destination)
        return filename

    def delete(self, img_url):
        fullpath = "/".join([UPLOAD_FOLDER, img_url])
        if os.path.exists(fullpath):
            os.remove(fullpath)
        else:
            logger.info("The file does not exist")


class img_handler_db(img_handler):
    """Handle images stored in MongoDB

    Args:
        img_handler ([type]): [description]
    """

    def load(self, img_url):
        pass

    def save(self, img):
        pass

    def save_file(self, file):
        pass

    def delete(self, img_url):
        pass


img_handler = img_handler_fs()

if __name__ == "__main__":
    mocked_imgs = ["2ab56008-88b3-41df-9b14-0176cc49e020.jpg", "16a1f729-2be3-4b13-9a3b-843c971dfa14.jpg"]
    for url in mocked_imgs:
        img = img_handler.load(url)
        cv2.imshow("test image", img)
        print(img_handler.save(img))
        cv2.waitKey(0)
