import matplotlib.pyplot as plt

N = []
for i in range(18, 28, 2):
    xlabel = "$2^{"
    xlabel += str(i)
    xlabel += '}$'
    N.append(xlabel)
# N = [r"$2^{}$".format(i) for i in range(18, 28, 2)]

# Drawing increasing boids, with visuals
cpu = [0.1506, 0.5339, 2.3771, 13.9042, 56.4294]
gpu_naive = [0.106496,  0.754688,  3.23482, 14.9873, 63.2361]
gpu_efficient = [0.157696, 0.443392, 1.40288, 5.29101, 20.8722]
gpu_thrust = [0.160768, 0.18832, 0.319488, 0.975808, 2.20467]
plt.xlabel("N count")
plt.ylabel("Time (ms)")
plt.plot(N, cpu, label="CPU", color="r", marker="o")
plt.plot(N, gpu_naive, label="Naive GPU", color="g", marker="o")
plt.plot(N, gpu_efficient, label="Efficient GPU", color="b", marker="o")
plt.plot(N, gpu_thrust, label="Thrust", color="y", marker="o")
plt.legend(labels=["CPU", "Naive GPU", "Efficient GPU", "Thrust"])
plt.title("Time of scan using different method(power-of-two)")
plt.show()
plt.savefig("../img/scan-power-of-two.png")
plt.clf()

cpu = [0.1685, 0.6555, 3.2271, 17.4222, 44.574]
gpu_naive = [0.1024, 0.765952, 3.24301, 15.574, 74.4376]
gpu_efficient = [0.166912, 0.477184, 1.41312, 6.2505, 20.8456]
gpu_thrust = [0.166944, 0.191584, 0.401504, 0.777952, 2.18931]
plt.xlabel("N count")
plt.ylabel("Time (ms)")
plt.plot(N, cpu, label="CPU", color="r", marker="o")
plt.plot(N, gpu_naive, label="Naive GPU", color="g", marker="o")
plt.plot(N, gpu_efficient, label="Efficient GPU", color="b", marker="o")
plt.plot(N, gpu_thrust, label="Thrust", color="y", marker="o")
plt.legend(labels=["CPU", "Naive GPU", "Efficient GPU", "Thrust"])
plt.title("Time of scan using different method(non-power-of-two)")
plt.show()
plt.savefig("../img/scan-non-power-of-two.png")
plt.clf()

exit()

# Drawing increasing boids, no visuals
fps_brutalForce = [8297.48, 1227.13, 42.882, 0.46393, 0.0]
fps_scatteredGrid = [7666.14, 7279.71, 4348.93, 380.379, 18.6696]
fps_coherentGrid = [7383.36, 7034.57, 4743.67, 683.372, 57.848]

plt.xlabel("N count")
plt.ylabel("FPS")
plt.plot(N, fps_brutalForce, label="brutalForce", color="r", marker="o")
plt.plot(N, fps_scatteredGrid, label="scatteredGrid", color="g", marker="o")
plt.plot(N, fps_coherentGrid, label="coherentGrid", color="b", marker="o")
plt.legend(labels=["brutalForce", "scatteredGrid", "coherentGrid"])
plt.title("FPS of kernel under different method (No visulization)")
# plt.show()
plt.savefig("../images/boids_no_visual.png")

plt.clf()

# Drawing 27-cell vs 8-cell
plt.xlabel("N count")
plt.ylabel("FPS")
fps_scatteredGrid27Neighbors = [7285.03, 6472.57, 3955.12, 353.385, 18.667]
fps_scatteredGrid8Neighbors = [7666.14, 7279.71, 4348.93, 380.379, 18.6696]
plt.plot(N, fps_scatteredGrid27Neighbors, label="27-cell", marker="o")
plt.plot(N, fps_scatteredGrid8Neighbors, label="8-cell", marker="o")
plt.legend(labels=["27-cell", "8-cell"])
plt.title("FPS of 27-cell vs 8-cell (No visualization)")
# plt.show()
plt.savefig("../images/27vs8_no_visual.png")

plt.clf()


# Drawing increasing blockSize

blockSize = [r"$2^{}$".format(i) for i in range(2, 11)]
fps_coherentGridBlockSizeChanged = [ 11.7512, 18.7417,28.524,41.9919, 57.0016, 62.4268, 64.3508, 61.0085, 59.5032]
plt.xlabel("Block size")
plt.ylabel("FPS")
plt.plot(blockSize, fps_coherentGridBlockSizeChanged, marker="o")
plt.title("FPS of coherent grid over increasing grid size (With visualization)")
# plt.show()
plt.savefig("../images/increasing_gridSize.png")

if __name__ == "__main__":
    pass