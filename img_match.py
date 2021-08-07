import numpy as np
import cv2
import os


def CorrectImage(queryImagePath, templateImgDir, outImg, val_num=100, threshold=90):
    """
    Find the template class for the query image and to correct the query image
    :param queryImagePath: the path of the query image  eg. "./img_test/test3.png"
    :param templateImgDir: the dir of the template      eg. "./template/"
    :param outImg:  the out put dir of corrected image  eg. "./img_test_corrected/"
    :param val_num: the number of samples for validating the Homography matrix eg. val_num=100
    :param threshold: the error threshold of the Homography mapping
                      from A to B using Homography_matrix. Suggest:
                      30<= threshold <=100 (Note that the results we got after the experiment
                      by Statistically matching template mean)
    :return the class of template or None
    """

    # queryImage
    queryImage = cv2.imread(queryImagePath, 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors of queryImage with SIFT
    queryImageKP, queryImageDES = sift.detectAndCompute(queryImage, None)

    # template images
    result = []  # [{"template_class": 1, "template_filename": "class1.png",
    # "homography_matrix": narray() }]
    for templateImgName in os.listdir(templateImgDir):
        # get the keypoints and descriptors of templateImage with SIFT
        templateImgPath = templateImgDir + templateImgName
        templateImg = cv2.imread(templateImgPath, 0)
        templateImgKP, templateImgDES = sift.detectAndCompute(templateImg, None)

        # match the keypoints
        bfMatcher = cv2.BFMatcher(crossCheck=True)
        matches = bfMatcher.match(queryImageDES, templateImgDES)
        matchesSorted = sorted(matches, key=lambda x: x.distance)

        """
        choose the first four matches to compute the Homography matrix
        and other 100 keypoints to validate the Homography matrix.
        """
        matchesForHMatrix = matchesSorted[:4]
        matchesForValidateH = matchesSorted[4:4 + val_num]

        # get the Homography matrix
        src_points = []
        target_points = []
        for match in matchesForHMatrix:
            query_index = match.queryIdx
            src_points.append(queryImageKP[query_index].pt)
            template_index = match.trainIdx
            target_points.append(templateImgKP[template_index].pt)
        hMatrix, s = cv2.findHomography(np.float32(src_points), np.float32(target_points), cv2.RANSAC, 10)

        # statistical the val set to find matching points to compute
        # the ratio of suitability
        error_total = 0
        for valMatche in matchesForValidateH:
            valsrc_index = valMatche.queryIdx
            valsrc_point = queryImageKP[valsrc_index].pt
            valsrc_point = valsrc_point + (1,)
            valtarget_index = valMatche.trainIdx
            valtarget_point = templateImgKP[valtarget_index].pt
            valtarget_point = valtarget_point + (1,)
            valsrc_point = np.array(valsrc_point)
            valtarget_point = np.array(valtarget_point)

            # b = H * aT
            error = np.sum(np.abs(valtarget_point - np.matmul(hMatrix, valsrc_point)))
            error_total = error_total + error

        if error_total / val_num < threshold:  # maybe change the threshold
            # finded the right template
            template_finded = {"template_class": int(templateImgName.split(".")[0][5:]),
                               "template_filename": templateImgName,
                               "homography_matrix": hMatrix}
            result.append(template_finded)
        # Draw first 10 matches.
        # imgShow = cv2.drawMatches(queryImage, queryImageKP, templateImg,
        #                          templateImgKP, matchesSorted[:10], None, flags=2)
        # plt.imshow(imgShow), plt.show()
        # cv2.findHomography()

    if len(result) == 0:
        print("no find the correct template")
        return None
    if len(result) > 1:
        print("warring: there are two templates that match the query image and we just return one")

    # template class
    result_tamplate_class = result[0]["template_class"]

    # correct the query img
    corrected_img = cv2.warpPerspective(queryImage, result[0]["homography_matrix"], queryImage.shape)
    cv2.imwrite(outImg + queryImagePath.split("/")[-1], corrected_img)
    
    return result_tamplate_class
if __name__ == "__main__":
    queryImagePath = "./img_test/test1.png"     # the image to be corrected
    templateImgDir = "./template/"      # the tamplate dir
    outImg = "./img_test_corrected/"

    # find the corresponding template and correct the img
    matchedTemplateClass = CorrectImage(queryImagePath, templateImgDir, outImg)