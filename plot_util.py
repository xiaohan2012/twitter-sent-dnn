import numpy as np
from matplotlib import pyplot as plt
def plot_track(means, stds, title):
    """
    Track means and stds as described in http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot the activation history
    layer_i = 1
    assert len(means[0]) > 0
    colors = ["b", "g", "y"]
    
    assert len(colors) > len(means)
    
    for ms, std, color in zip(means, stds, colors):
        ms, std = np.array(ms), np.array(std)
        
        ax.plot(ms, color+'o-', label = "Layer %d" %(layer_i))                
        ax.hold(True)
        
        #upper and lower
        ax.plot(ms+std, color+'*-')
        ax.plot(ms-std, color+'*-')
        
        layer_i += 1
        
    ax.set_title(title)
    ax.legend(loc='best', fancybox=True)

def plot_hist(rows, title):
    """
    histogram plot as described in http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot the activation history
    layer_i = 1
    assert len(rows[0]) > 0
    for x in rows:
        ax.hist(x, 
                bins = 100, 
                label = "Layer %d" %(layer_i), 
                normed = True, 
                histtype = "step"
        )
        ax.hold(True)
        
        layer_i += 1

    ax.set_title(title)
    ax.legend(loc='best', fancybox = True)
