import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.interpolate import griddata
from skimage.draw import polygon
import argparse

# Pick points to align on both images
def pick_pts(im, N=4):
    plt.imshow(im)
    x_coors = []
    y_coors = []
    
    for i in range(N):
        x, y = plt.ginput(1)[0]
        x_coors.append(x)
        y_coors.append(y)
        
        plt.close()
        plt.plot(x_coors, y_coors, '.r')
        plt.annotate(i+1, (x,y), xytext=(x+50, y+50))
        
        plt.imshow(im)
        
    plt.close()
    return np.stack([x_coors, y_coors], axis=1)


def compute_affine(im1, im1_pts, im2_pts, tri, g, size):
    #A*Tri1 = Tri2
    assert im1_pts.shape == im2_pts.shape
    
    #triangulation on im2_pts (Delaunay on avg pts)
    im2_tri = tri(im2_pts)
    
    morphed_x = []
    morphed_y = []
    m = []

    for t in im2_tri.simplices:

        #get im1 triangle
        im1_t = im1_pts[t]
        im1_t_trans = np.transpose(im1_t)
        
        #get im2 triangle
        im2_t = im2_pts[t]
        im2_t_trans = np.transpose(im2_t)
           
       
        #matrix that transforms identity to triangle
        A1 = tri_to_identity(im1_t_trans[0], im1_t_trans[1])
        A2 = tri_to_identity(im2_t_trans[0], im2_t_trans[1])
        
        #cc1 are x pixels, rr1 are y pixels
        rr1, cc1 = polygon(im1_t_trans[1], im1_t_trans[0])
        t1_pix = np.vstack((cc1, rr1, np.ones_like(rr1)))
        
        
        #transform t1 pixels to identity
        id_t1 = np.linalg.lstsq(A1, t1_pix, rcond=None)[0]
        
        #transform t1_identity to t2 shape
        t1_avg = np.intp(np.matmul(A2, id_t1))
        
        #column and row pixels of transformed
        #x pixels
        mc1 = t1_avg[0]
        #y pixels
        mr1 = t1_avg[1]

        
        morphed_x.extend(mc1)
        morphed_y.extend(mr1)
        m.extend(im1[rr1, cc1].tolist())
        
    interpolated = interpolate((morphed_x, morphed_y), np.array(m), g, size)
       
    return interpolated

# Avg shape
def compute_avg_img(im1_pts, im2_pts, warp_frac):
    return (im1_pts*warp_frac) + (im2_pts*(1-warp_frac))

# affine warp for each triangle in the triangulation from the original images into this new shape
def affine_matrices(tris_1, tris_2):
    A = []
    for i in range(len(tris_1)):
        A.append(compute_affine(tris_1[i], tris_2[i]))
        
    return A

def tri_to_identity(tri_x, tri_y):
    """The matrix transforming a triangle's vertices to the identity triangle:
    ((0,0), (0,1), (1,0))"""

    a_x, b_x, c_x = tri_x
    a_y, b_y, c_y = tri_y
    return np.intp(np.array([
        [c_x - a_x, b_x - a_x, a_x],
        [c_y - a_y, b_y - a_y, a_y],
        [0,         0,         1]
    ]))

def morph_full(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    
    #get avg points
    avg_pts = compute_avg_img(im1_pts, im2_pts, warp_frac)
    
    #interpolation grid
    ht = im1.shape[0]
    wt = im1.shape[1]
    g = np.array([[w, h] for h in np.arange(ht) for w in np.arange(wt)])

    size = (ht, wt)
    
    #Compute affine matrix and morph images to avg
    im1_av = compute_affine(im1, im1_pts, avg_pts, tri, g, size)

    im2_av = compute_affine(im2, im2_pts, avg_pts, tri, g, size)

    morphed_full = np.round(im1_av*dissolve_frac + im2_av*(1-dissolve_frac)).astype('uint8')
    
    return np.clip(morphed_full, 0, 255)

def interpolate(points, values, grid, size):

    #seperate into rgb channels and interpolate
    r = values[:, 0]
    g = values[:, 1]
    b = values[:, 2]
    
    r_interp = griddata(points, r, grid, method='nearest').reshape(size)
    g_interp = griddata(points, g, grid, method='nearest').reshape(size)
    b_interp = griddata(points, b, grid, method='nearest').reshape(size)
    
    merge = np.dstack((r_interp, g_interp, b_interp))

    return merge

def main(p1_path, p2_path):
    print("in main")
    p1_img = skio.imread(p1_path)
    p2_img = skio.imread(p2_path)

    p1_pts = pick_pts(p1_img, 44)
    p2_pts = pick_pts(p2_img, 44)
    frac = np.linspace(0, 1, 45)
    for t in range(45):
        morph = morph_full(p1_img, p2_img, p1_pts, p2_pts, Delaunay, frac[t], frac[t])
        plt.imsave('morph_gif_' + str(t) + '_.jpeg', morph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #print("top level")
    parser.add_argument('im1_path', type=str)
    parser.add_argument('im2_path', type=str)

    args = parser.parse_args()

    p1_path = args.im1_path
    p2_path = args.im2_path
    main(p1_path, p2_path)