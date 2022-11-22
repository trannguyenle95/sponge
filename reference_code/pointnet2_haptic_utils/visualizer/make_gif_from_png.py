# make a gif from the stored images
import imageio

robot_name = 'sawyer'
images = []
for _ in range(51, 250):
    filename = "data/{}-contact-force-visual-{}-force.png".format(robot_name, _)
    images.append(imageio.imread(filename))
imageio.mimsave('data/{}-contact-visual-force.gif'.format(robot_name), images)