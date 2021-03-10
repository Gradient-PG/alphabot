import gym_line_follower
import gym
import time


if __name__ == "__main__":
    env = gym.make("LineFollower-v0")
    env.reset()
    for _ in range(100):
        for i in range(1000):
            obsv, rew, done, info = env.step((0.0, 0.0))
            time.sleep(0.05)
            if done:
                break
        env.reset()
    env.close()
