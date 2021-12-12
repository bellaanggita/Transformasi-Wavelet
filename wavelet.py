import cv2
import numpy
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

#import gambar
culvularia = cv2.imread("culvularia-patogen.png")
fusarium = cv2.imread("fusarium-patogen.png")

#konversi color space ke hsv, ycc, dan grayscale
#culvularia
culvularia_HSV = cv2.cvtColor(culvularia,cv2.COLOR_BGR2HSV)
culvularia_YCC = cv2.cvtColor(culvularia,cv2.COLOR_BGR2YCrCb)
culvularia_Grayscale = cv2.cvtColor(culvularia,cv2.COLOR_BGR2GRAY)

#fusarium
fusarium_HSV = cv2.cvtColor(fusarium,cv2.COLOR_BGR2HSV)
fusarium_YCC = cv2.cvtColor(fusarium,cv2.COLOR_BGR2YCrCb)
fusarium_Grayscale = cv2.cvtColor(fusarium,cv2.COLOR_BGR2GRAY)

#split color space
cul_H, cul_S, cul_V = cv2.split(culvularia_HSV)
cul_Y, cul_Cr, cul_Cb = cv2.split(culvularia_YCC)

fus_H, fus_S, fus_V = cv2.split(fusarium_HSV)
fus_Y, fus_Cr, fus_Cb = cv2.split(fusarium_YCC)

#thresholding pathogen culvaria
nilai_cul = filters.threshold_otsu(culvularia_Grayscale)

hist, bins_center = exposure.histogram(culvularia_Grayscale)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(culvularia_Grayscale, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(culvularia_Grayscale < nilai_cul, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(nilai_cul, color='k', ls='--')

#menampilkan plot hasil thresholding pathogen culvaria
plt.tight_layout()
plt.show()

#thresholding pathogen fusarium
nilai_fus = filters.threshold_otsu(fusarium_Grayscale)

hist2, bins_center2 = exposure.histogram(fusarium_Grayscale)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(fusarium_Grayscale, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(fusarium_Grayscale < nilai_fus, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center2, hist2, lw=2)
plt.axvline(nilai_fus, color='k', ls='--')

#menampilkan plot hasil thresholding pathogen fusarium
plt.tight_layout()
plt.show()


##wavelet pathogen culvularia
#1. segmentasi menggunakan otsu thresholding
row, col, ch = culvularia.shape
canvas_cuvularia = numpy.zeros((row,col), numpy.uint8)

for i in range (row):
    for j in range (col):
        if(culvularia_Grayscale[i,j]<nilai_cul):
            canvas_cuvularia.itemset((i,j), culvularia_Grayscale[i,j])

#2. dekomposisi 1 level dan 2 level
max_lev = 2
label_levels = 2

fig, axes = plt.subplots(2, 3, figsize=[14, 8])
for level in range(0, max_lev + 1):
    if level == 0:
        # menampilkan gambar original sebelum dekomposisi
        axes[0, 0].set_axis_off()
        axes[1, 0].imshow(canvas_cuvularia, cmap=plt.cm.gray)
        axes[1, 0].set_title('Image')
        axes[1, 0].set_axis_off()
        continue

    draw_2d_wp_basis((row,col,1), wavedec2_keys(level), ax=axes[0, level],
                     label_levels=label_levels)
    axes[0, level].set_title('{} level\ndecomposition'.format(level))

    # compute the 2D DWT
    c = pywt.wavedec2(canvas_cuvularia, 'db2', mode='periodization', level=level)
    c[0] /= numpy.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/numpy.abs(d).max() for d in c[detail_level + 1]]
    # menampilkan normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[1, level].imshow(arr, cmap=plt.cm.gray)
    axes[1, level].set_title('Coefficients\n({} level)'.format(level))
    axes[1, level].set_axis_off()

plt.tight_layout()
plt.show()

#3. dekomposisi per sumbu (x,y,diagonal)
titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(canvas_cuvularia, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()


##wavelet pathogen fusarium
#1. segmentasi menggunakan otsu thresholding
row, col, ch = fusarium.shape
canvas_fusarium = numpy.zeros((row,col), numpy.uint8)

for i in range (row):
    for j in range (col):
        if(fusarium_Grayscale[i,j]<nilai_cul):
            canvas_fusarium.itemset((i,j), fusarium_Grayscale[i,j])

#2. dekomposisi 1 level dan 2 level
max_lev = 2
label_levels = 2

fig, axes = plt.subplots(2, 3, figsize=[14, 8])
for level in range(0, max_lev + 1):
    if level == 0:
        # menampilkan gambar original sebelum dekomposisi
        axes[0, 0].set_axis_off()
        axes[1, 0].imshow(canvas_fusarium, cmap=plt.cm.gray)
        axes[1, 0].set_title('Image')
        axes[1, 0].set_axis_off()
        continue

    draw_2d_wp_basis((row,col,1), wavedec2_keys(level), ax=axes[0, level],
                     label_levels=label_levels)
    axes[0, level].set_title('{} level\ndecomposition'.format(level))

    # compute the 2D DWT
    c = pywt.wavedec2(canvas_fusarium, 'db2', mode='periodization', level=level)
    c[0] /= numpy.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/numpy.abs(d).max() for d in c[detail_level + 1]]
    # menampilkan normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[1, level].imshow(arr, cmap=plt.cm.gray)
    axes[1, level].set_title('Coefficients\n({} level)'.format(level))
    axes[1, level].set_axis_off()

plt.tight_layout()
plt.show()

#3. dekomposisi per sumbu (x,y,diagonal)
titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(canvas_fusarium, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#menampilkan hasil segmentasi pathogen culvularia dan fusarium
cv2.imshow("Segmentasi Pathogen Culvularia", cv2.resize(canvas_cuvularia,(800,600)))
cv2.imshow("Segmentasi Pathogen Fusarium", cv2.resize(canvas_fusarium,(800,600)))

#exit
k=cv2.waitKey()
#membuat file baru
if k == ord('s'):
    cv2.imwrite("Segmentasi Pathogen Culvularia.png", cv2.resize(canvas_cuvularia,(800,600)))
    cv2.imwrite("Segmentasi Pathogen Fusarium.png", cv2.resize(canvas_fusarium,(800,600)))
cv2.destroyAllWindows()
