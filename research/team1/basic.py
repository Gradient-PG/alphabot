import gym_line_follower
import gym
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    env = gym.make("LineFollower-v0")
    env.reset()
    total_reward, timestep = 0, 0
    action = [0, 0]
    done = False
    while not done:
        timestep += 1
        obsv, rew, done, info = env.step(action)
        total_reward += rew
        x, y = obsv[0], obsv[1]
        # action = [1 if y < 0 else 0, 1 if y > 0 else 0]
        action = [1 if y < 0 else 1 - y * 50, 1 if y > 0 else 1 + y * 50]
        logger.info(f"timestep: {timestep}\tx={x}, y={y}")

    logger.info(f"Total reward = {total_reward}")
    env.close()
