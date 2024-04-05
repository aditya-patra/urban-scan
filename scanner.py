import cv2
import tensorflow as tf
'''
list of env
dict env to colors
3-d array for storing environments
for each pixel, check if pixels around it are from same env
if from same env:
    if pixel is already part of environment, join environment
    else create new environment
'''
#use PNG files as lossy does not apply
img = cv2.imread('satellite.png')
cv2.imwrite('satellite_copy.png', img)
img = cv2.resize(img, (100, 100))
cv2.imwrite('satellite_resize.png', img)
cv2.imshow('image', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
height, width, c = img.shape
#stores RGB values of each pixel
pixel_arr = []
#stores env type of each pixel
env_arr = []
#stores list of environments with corresponding pixel locations
env_list = []
#stores env type of pixels being urbanized
urban_arr = []
#checks for water
warning_water = 0
#checks for percent land destroyed

print(height, width)
counter = 0
outline_x = [width, -1]
outline_y = [height, -1]
inputted = input('Enter a comma seperated list of environments: ')
env = ['forest', 'grassland', 'ocean']
env = inputted.split(", ")
for i, environment in enumerate(env):
    while len(env[i]) < 9:
        env[i] = env[i] + " "
#size of image input in km
height_km = int(input("Enter height of img(km): "))
width_km = int(input("Enter width of img(km): "))
#size of each pixel
pixel_width = int(width_km/100)
pixel_height = int(height_km/100)
env_sqr_km = int(25/(pixel_width*pixel_height))
env_dict = {
    'g': ['forest   '],
    'lg':['forest   '],
    'o': ['grassland'],
    'y': ['grassland'],
    'c': ['ocean    '],
    'blue': ['ocean    '],
    'w': ['ocean    '],
    'b': ['x        '],
    'r': ['outline  '],
    'x': ['x        '],
    'outline': ['outline  ']
}
for environment in env:
    if environment == 'forest   ':
        if 'grassland' not in env:
            env_dict['g'] = ['forest   ']
            env_dict['lg'] = ['forest   ']
        else:
            env_dict['g'] = ['forest   ']
    elif environment == 'grassland':
        if 'forest   ' not in env:
            env_dict['g'] = ['grassland']
            env_dict['lg'] = ['grassland']
        else:
            env_dict['lg'] = ['grassland']
    elif environment == 'savanna  ':
        if 'desert   ' not in env:
            env_dict['o'] = ['savanna  ']
            env_dict['y'] = ['savanna  ']
        else:
            env_dict['y'] = ['savanna  ']
    elif environment == 'desert   ':
        if 'savanna  ' not in env:
            env_dict['o'] = ['desert   ']
            env_dict['y'] = ['desert   ']
        else:
            env_dict['o'] = ['desert   ']
    elif environment == 'tundra   ':
        if 'ocean    ' not in env and 'shallow water' not in env:
            env_dict['c'] = ['tundra   ']
            env_dict['blue'] = ['tundra   ']
            env_dict['w'] = ['tundra   ']
        elif 'shallow water' not in env:
            env_dict['c'] = ['tundra   ']
            env_dict['w'] = ['tundra   ']
        elif 'ocean    ' not in env:
            env_dict['w'] = ['tundra   ']
            env_dict['blue'] = ['tundra   ']
        else:
            env_dict['w'] = ['tundra   ']
    elif environment == 'ocean    ':
        if 'tundra   ' not in env and 'shallow water' not in env:
            env_dict['c'] = ['ocean    ']
            env_dict['blue'] = ['ocean    ']
            env_dict['w'] = ['ocean    ']
        elif 'shallow water' not in env:
            env_dict['c'] = ['ocean    ']
            env_dict['blue'] = ['ocean    ']
        elif 'tundra   ' not in env:
            env_dict['w'] = ['ocean    ']
            env_dict['blue'] = ['ocean    ']
        else:
            env_dict['blue'] = ['ocean    ']
    elif environment == 'shallow water':
        if 'ocean    ' not in env and 'tundra   ' not in env:
            env_dict['w'] = ['shallow water']
            env_dict['c'] = ['shallow water']
            env_dict['blue'] = ['shallow water']
        elif 'tundra   ' not in env:
            env_dict['c'] = ['shallow water']
            env_dict['w'] = ['shallow water']
        elif 'ocean    ' not in env:
            env_dict['c'] = ['shallow water']
            env_dict['blue'] = ['shallow water']
        else:
            env_dict['c'] = ['shallow water']
print(env_dict)
'''
ice - i
forest - f
grassland - g
desert - d
deep water - dw
shallow water - sw
savanna - s
dirt - b
urban/suburban - x
farmland - fl
Water Bodies: (0, 0, 100) to (0, 100, 255)
Forests: (0, 100, 0) to (50, 150, 50)
Grasslands/Savannas: (100, 150, 0) to (200, 200, 100)
Deserts: (200, 150, 50) to (255, 200, 100)
Snow/Ice: (200, 200, 200) to (255, 255, 255)
Urban/Man-made: (100, 100, 100) to (200, 200, 200)
Agricultural Lands: (150, 100, 50) to (200, 150, 100)
Wetlands: (0, 100, 100) to (0, 150, 150)
'''
model = tf.keras.models.load_model("checkpoint_path")
for y in range(height):
    row = []
    env_row = []
    for x in range(width):
        pixel = img[y][x]
        #cv2.imshow('image', pixel)
        #cv2.waitKey(20)
        #cv2.destroyAllWindows()
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]
        #print([r, g, b])
        separated_image = pixel.reshape(1, 1, 3)
        resized_image = cv2.resize(separated_image, (150, 150))
        resized_image = resized_image.reshape(1, 150, 150, 3)
        prediction = model.predict(resized_image)
        print(prediction)
        #if str(prediction) == "[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]":
            #row.append('outline')
            #if x > outline_x[0]:
                #outline_x[0] = x
            #if outline_y[0] < y:
                #outline_y[0] = y
            #if x > outline_x[1]:
                #outline_x[1] = x
            #if outline_y[1] > y:
                #outline_y[1] = y
        if str(prediction) == "[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]":
            row.append('w')
            env_row.append(env_dict['w'])
        elif str(prediction) == "[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]":
            row.append('b')
            env_row.append(env_dict['b'])
        elif str(prediction) == "[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]":
            row.append('r')
            env_row.append(env_dict['r'])
        elif (str(prediction) == "[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]" or str(prediction) == "[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]") and g < 200:
            row.append('g')
            env_row.append(env_dict['g'])
        elif (str(prediction) == "[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]" or str(prediction) == "[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]") and g >= 200:
            row.append('lg')
            env_row.append(env_dict['lg'])
        elif str(prediction) == "[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]":
            row.append('blue')
            env_row.append(env_dict['blue'])
        elif str(prediction) == "[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]":
            row.append('c')
            env_row.append(env_dict['c'])
        elif str(prediction) == "[[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]":
            row.append('y')
            env_row.append(env_dict['y'])
        elif str(prediction) == "[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]":
            row.append('o')
            env_row.append(env_dict['o'])
        else:
            row.append('x')
            env_row.append(env_dict['x'])
    pixel_arr.append(row)
    env_arr.append(env_row)
    counter += 1
f = open('img_env.txt', 'w')
for row in env_arr:
    f.write(str(row)+"\n")
f.close()
#print(env_arr)
#print(pixel_arr)
#print(width)
#print(height)
#print(len(pixel_arr[0]))
#print(len(pixel_arr))
for y in range(height):
    for x in range(width):
        new_group = 1
        #print('b')
        #print([x, y])
        #print(pixel_arr[x][y])
        #print(env_dict[pixel_arr[x][y]])
        #print(pixel_arr[x][y+1])
        if pixel_arr[y][x] == 'x':
            continue
        if x > 0 and y > 0:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y-1][x-1]]:
                for i, group in enumerate(env_list):
                    if [y-1, x-1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if x > 0 and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y][x-1]]:
                for i, group in enumerate(env_list):
                    if [y, x-1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if y < (height-1) and x > 0 and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y+1][x-1]]:
                for i, group in enumerate(env_list):
                    if [y+1, x-1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if x < (width-1) and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y][x+1]]:
                for i, group in enumerate(env_list):
                    if [y, x+1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if y > 0 and x < (width-1) and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y-1][x+1]]:
                for i, group in enumerate(env_list):
                    if [y-1, x+1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if y < (height-1) and x < (width-1) and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y+1][x+1]]:
                for i, group in enumerate(env_list):
                    if [y+1, x+1] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if y > 0 and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y-1][x]]:
                for i, group in enumerate(env_list):
                    if [y-1, x] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if y < (height-1) and new_group == 1:
            if env_dict[pixel_arr[y][x]] == env_dict[pixel_arr[y+1][x]]:
                for i, group in enumerate(env_list):
                    if [y+1, x] in group:
                        new_group = 0
                        env_list[i].append([y,x])
                        break
        if new_group == 1:
            #list_to_append = [env_dict[pixel_arr[x, y]], [x, y]]
            #print(type(list_to_append))
            env_list.append([[y, x]])
#print('b')
to_be_popped = []
for i, environment in enumerate(env_list):
    if len(environment) < env_sqr_km:
        to_be_popped.append(i)
#print(to_be_popped)
to_be_popped.reverse()
#print(to_be_popped)
#print(len(env_list))
for index in to_be_popped:
    env_list.pop(index)
#print(len(env_list))
#print(env_list[0])
img_edited = cv2.imread('satellite_edited.png', cv2.IMREAD_UNCHANGED)
#img_edited = cv2.cvtColor(img_edited, cv2.COLOR_BGR2RGB)
cv2.imwrite('sat_edit_resize.png', img_edited)
cv2.imshow('image', img_edited)
cv2.waitKey(1000)
cv2.destroyAllWindows()
for y in range(height):
    for x in range(width):
        pixel = img_edited[y, x]
        #print(img_edited[y, x])
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]
        #print([r, g, b])
        #if r > 170 and (g < 40 or b < 40):
            #print([r, g, b])
        if r > 170 and g < 40 and b < 40:
            #print('r')
            if x < outline_x[0]:
                outline_x[0] = x
            elif x > outline_x[1]:
                outline_x[1] = x
            if y < outline_y[0]:
                outline_y[0] = y
            elif y > outline_y[1]:
                outline_y[1] = y
#print(outline_x)
#print(outline_y)
num_destroyed = []
num_in_img = []
for environment in env:
    num_destroyed.append(0)
    num_in_img.append(0)
print(env_arr[0])
for y in range(outline_y[0], outline_y[1]):
    row_destroyed = env_arr[y]
    part_destroyed = row_destroyed[outline_x[0]: outline_x[1]]
    urban_arr.append(part_destroyed)
    if ['ocean    '] in part_destroyed:
        warning_water = 1
    for i, environment in enumerate(env):
        num_destroyed[i] += part_destroyed.count([environment])
print(num_destroyed)
for y in range(height):
    for i, environment in enumerate(env):
        num_in_img[i] += env_arr[y].count([environment])
print(env)
print(num_in_img)
print(outline_y)
print(outline_x)
environments_destroyed = []
environments_destroyed_num = 0
env_destroyed_list = []
for environment in env_list:
    portion_destroyed = 0
    for location in environment:
        if location[0] >= outline_y[0] and location[0] <= outline_y[1] and location[1] >= outline_x[0] and location[1] <= outline_x[1]:
            portion_destroyed += 1
    if portion_destroyed > len(environment)/4:
        environments_destroyed.append(env_arr[environment[0][1]][environment[0][0]])
        environments_destroyed_num += 1
        env_destroyed_list.append(environment)
print("- - - - - - - - - - - -")
print("Detected "+str(len(env_list))+" different environments at least 25 square km large")
#print(environments_destroyed)
print("At least 25% of "+str(environments_destroyed_num)+" were destroyed")
if environments_destroyed_num > 0:
    print("Make sure to relocate any animals in these environments")
if warning_water == 1:
    print("Detected water in area about to be urbanized")
for i, environment in enumerate(env):
    print(environment.strip(" ")+": "+str(num_destroyed[i])+" square km destroyed")
    if num_destroyed[i] > num_in_img[i]/4:
        print("This urbanization is unsafe for the wildlife in the area due to too much of the "+environment.strip(" ")+" being destroyed")
print("Here is an updated version of the image showing large affected environments highlighted in red: ")
for environment in env_list:
    for point in environment:
        img[point[0], point[1]] = (255, 0, 255)
for environment in env_destroyed_list:
    for point in environment:
        img[point[0], point[1]] = (0, 0, 255)
cv2.imshow('image', img)
cv2.waitKey(5000)
cv2.imwrite('satellite_highlighted.png', img)