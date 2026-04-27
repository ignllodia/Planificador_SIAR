import cv2

# Cargar imagen
img = cv2.imread('sewer_map.png', cv2.IMREAD_GRAYSCALE)

# Escalar por factor 3 usando nearest neighbor
img_resized = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

# Guardar resultado
cv2.imwrite('sewer_map_x3.png', img_resized)



