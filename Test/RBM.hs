module Test.RBM(perf
               ,test
               ) where

--local
import qualified Data.RBM as R
import qualified Data.DNN.Trainer as T
import qualified Data.Matrix as M

import Data.Matrix((-^),
                   Matrix(..),
                   U,
                   B,
                   I)

--utils
import qualified System.Random as Rnd
import Control.Monad.Identity(runIdentity)
import Control.Monad(forever,when)

--benchmark modules
import Criterion.Main(defaultMainWith,defaultConfig,bgroup,bench,whnf)
import Criterion.Types(reportFile,timeLimit)

--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)

import Debug.Trace(traceShowId)

seeds :: Int -> [Int] 
seeds seed = Rnd.randoms (Rnd.mkStdGen seed)

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

finishIf :: Monad m => Int -> Double -> Matrix U B I -> T.Trainer m ()
finishIf n e b = do 
   cnt <- T.getCount
   when (n < cnt) T.finish_
   err <- T.reconErr b
   when (e > err) T.finish_

-- |test to see if we can learn a random string
prop_learn :: Word8 -> Word8 -> Word8 -> Bool
prop_learn bs ni nh = runIdentity $ do
   let rbm = R.new s1 (fi ni) (fi nh)
       (s1:s2:_) = seeds $ (fi ni) * (fi nh) * (fi bs)
       fi ww = 3 + (fromIntegral ww)
       toD = fromIntegral :: (Int -> Double)
       bits = take ((fi bs) * (fi ni)) $ map (toD . (`mod` 2)) $ seeds s2
       inputs = M.fromList (fi bs, fi ni) bits
   erst <- fst <$> (T.run [rbm] $ T.reconErr inputs)
   let train = do T.setLearnRate 0.25
                  finishIf 1000 erst inputs
                  T.contraDiv inputs
   lrb <- snd <$> (T.run [rbm] $ forever train)
   recon <- R.reconstruct inputs lrb
   err <- traceShowId <$> M.mse (inputs -^ recon)
   return $ (err < erst || err < 0.5)

-- |test to see if we fail to rearn with a negative learning rate
prop_not_learn :: Word8 -> Word8 -> Word8 -> Bool
prop_not_learn bs ni nh = runIdentity $ do
   let rbm = R.new s1 (fi ni) (fi nh) 
       fi ww = 3 + (fromIntegral ww)
       (s1:s2:_) = seeds $ (fi ni) * (fi nh) * (fi bs)
       toD = fromIntegral :: (Int -> Double)
       bits = take ((fi bs) * (fi ni)) $ map (toD . (`mod` 2)) $ seeds s2
       inputs = M.fromList (fi bs, fi ni) bits
       train = do T.setLearnRate (-0.25)
                  finishIf 100 0.05 inputs
                  T.contraDiv inputs
   erst <- fst <$> (T.run [rbm] $ T.reconErr inputs)
   lrb <- snd <$> (T.run [rbm] $ forever train)
   recon <- R.reconstruct inputs lrb
   err <- traceShowId <$> M.mse (inputs -^ recon)
   return $ (err >= erst || err >= 0.5)

prop_init :: Word8 -> Word8 -> Bool
prop_init ni nh = (fi ni) * (fi nh)  == (M.elems rb)
   where
      seed = (fi ni) * (fi nh)
      rb = R.new seed (fi ni) (fi nh)
      fi :: Word8 -> Int
      fi ww = 1 + (fromIntegral ww)

prop_hiddenProbs :: Word8 -> Word8 -> Bool
prop_hiddenProbs ni nh = runIdentity $ do
   let rb = R.new seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       input = M.randomish (1, (fi ni)) (0,1) seed
       seed = (fi ni) * (fi nh)
   pp <- R.hiddenPs rb input
   return $ (fi nh) == (M.col pp)

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = runIdentity $ do
   let --h0 = w00 * i0 + w10 * i1
       h1 = w01 * i0 + w11 * i1
       h2 = w02 * i0 + w12 * i1
       i0:i1:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       input = M.fromList (1,2) $ [i0,i1]
       rb = M.fromList (2,3) wws
   pp <- M.toList <$> R.hiddenPs rb input
   return $ pp == 1:(map sigmoid [h1, h2])

prop_inputProbs :: Word8 -> Word8 -> Bool
prop_inputProbs ni nh = runIdentity $ do
   let hidden = M.randomish (1,(fi nh)) (0,1) seed
       rb = R.new seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       seed = (fi ni) * (fi nh)
   pp <- R.inputPs rb hidden
   return $ (fi ni) == (M.row pp)

prop_inputProbs2 :: Bool
prop_inputProbs2 = runIdentity $ do
   let --i0 = w00 * h0 + w10 * h1
       i1 = w01 * h0 + w11 * h1
       i2 = w02 * h0 + w12 * h1
       h0:h1:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       hiddens = M.fromList (1,2) [h0,h1]
       rb = M.fromList (2,3) $ wws
   rb' <- M.transpose rb
   pp <- R.inputPs rb' hiddens
   pp' <- M.toList <$> M.transpose pp
   return $ pp' == 1:(map sigmoid [i1,i2])

prop_energy :: Word8 -> Word8 -> Bool
prop_energy ni nh = runIdentity $ do
   let input = M.randomish (1, (fi ni)) (0,1) seed
       rb = R.new seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       seed = (fi ni) * (fi nh)
   ee <- R.energy rb input
   return $ not $ isNaN ee

prop_recon :: Word8 -> Word8 -> Word8 -> Word8 -> Bool
prop_recon bs ni nh no = runIdentity $ do
   let input = M.randomish (fi bs, fi ni) (0,1) seed
       r1 = R.new seed (fi ni) (fi nh)
       r2 = R.new seed (fi nh) (fi no)
       fi ww = 1 + (fromIntegral ww)
       seed = (fi ni) * (fi nh)
   recons <- R.reconstruct input [r1,r2]
   return $ M.shape recons == M.shape input

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "init"      prop_init
   runtest "energy"    prop_energy
   runtest "hiddenp"   prop_hiddenProbs
   runtest "hiddenp2"  prop_hiddenProbs2
   runtest "inputp"    prop_inputProbs
   runtest "inputp2"   prop_inputProbs2
   runtest "recon"     prop_recon
   runtest "learn"     prop_learn
   runtest "not_learn" prop_not_learn

perf :: IO ()
perf = do
   let file = "dist/perf-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 1.0 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "63x63"  $ whnf (prop_energy 63) 63
                       , bench "127x127"  $ whnf (prop_energy 127) 127
                       , bench "255x255"  $ whnf (prop_energy 255) 255
                       ]
      ,bgroup "hidden" [ bench "63x63"  $ whnf (prop_hiddenProbs 63) 63
                       , bench "127x127"  $ whnf (prop_hiddenProbs 127) 127
                       , bench "255x255"  $ whnf (prop_hiddenProbs 255) 255
                       ]
      ,bgroup "input" [ bench "63x63"  $ whnf (prop_inputProbs 63) 63
                      , bench "127x127"  $ whnf (prop_inputProbs 127) 127
                      , bench "255x255"  $ whnf (prop_inputProbs 255) 255
                      ]
      ,bgroup "learn" [ bench "7"  $ whnf (prop_learn 7 7) 7
                      , bench "15"  $ whnf (prop_learn 15 15) 15
                      ]
      ]
   putStrLn $ "perf log written to " ++ file
