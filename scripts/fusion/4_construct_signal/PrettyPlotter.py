def pretty(plt):
   plt.gca().spines['right'].set_visible(False)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().xaxis.set_ticks_position('bottom')
   plt.gca().yaxis.set_ticks_position('left')
   box = plt.gca().get_position()
   plt.gca().set_position([box.x0, box.y0, box.width*0.8, box.height])
   plt.tick_params(labelsize=20)
   return

   # Note: To put legend outside of plot:
   # plt.legend([], loc='upper right', bbox_to_anchor=(1,0))
