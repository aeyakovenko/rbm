module RBM.Repa(rbm
               --,learn
               ,energy
               ,perf
               ,test
               ) where

--benchmark modules
import Criterion.Main(defaultMainWith,defaultConfig,bgroup,bench,whnf)
import Criterion.Types(reportFile,timeLimit)
--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
--impl modules
import Data.Array.Repa.Algorithms.Randomish(randomishDoubleArray)
import Data.Array.Repa(Array
                      ,U
                      ,DIM2
                      ,DIM1
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      )
import qualified Data.Array.Repa as R
import System.Random(RandomGen
                    ,random
                    ,mkStdGen
                    )

import Control.Monad.Identity(runIdentity)

data RBM = RBM { weights :: Array U DIM2 Double -- weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1
               }

numHidden :: RBM -> Int
numHidden rb = (row $ R.extent $ weights rb) 

numInputs :: RBM -> Int
numInputs rb = (col $ R.extent $ weights rb) 

-- :. is an infix constructor similar to : for lists
row :: DIM2 -> Int
row (Z :. r :. _) = r

col :: DIM2 -> Int
col (Z :. _ :. c) = c

pos :: DIM1 -> Int
pos (Z :. i ) = i

--create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = RBM nw
   where
      nw = randomishDoubleArray (Z :. nh :. ni) 0 1 (fst $ random r)

{-- 
 - given an rbm and a biased input array, generate the energy
 - should be: negate $ sumAll $ weights *^ (hidden `tensor` biased)
 - but everything is unrolled to experiment with Repa's parallelization
 --}
energy :: RBM -> Array U DIM1 Double -> Double
energy rb biased = negate ee
   where
      ee = runIdentity $ hhs `R.deepSeqArray` R.sumAllP $ R.fromFunction sh (computeEnergyMatrix wws biased hhs)
      wws = weights rb
      hhs = hiddenProbs rb biased
      sh = (Z :. nh :. ni)
      ni = numInputs rb
      nh = numHidden rb

{-# INLINE computeEnergyMatrix #-}
computeEnergyMatrix ::  Array U DIM2 Double -> Array U DIM1 Double -> Array U DIM1 Double -> DIM2 -> Double 
computeEnergyMatrix wws biased hhs sh = ww * ii * hh
   where
      rr = row sh 
      cc = col sh 
      ww = wws `R.index` sh
      ii = biased `R.index` (Z :. cc) 
      hh = hhs `R.index` (Z :. rr)

{--
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - map sigmoid $ weights `mmult` biased
 -
 - 
 --}
{-# INLINE hiddenProbs #-}
hiddenProbs :: RBM -> Array U DIM1 Double -> Array U DIM1 Double
hiddenProbs rb biased = R.computeUnboxedS $ R.fromFunction shape (computeHiddenProb wws biased)
   where
      wws = weights rb
      shape = (Z :. (numHidden rb))

{--
 - sigmoid of the dot product of the row
 --}
{-# INLINE computeHiddenProb #-}
computeHiddenProb :: Array U DIM2 Double -> Array U DIM1 Double -> DIM1 -> Double
computeHiddenProb wws biased xi = sigmoid sm
   where
      rw = R.slice wws (Any :. (pos xi) :. All)
      sm = R.sumAllS $ R.zipWith (*) rw biased

{--
 - given a biased hidden sample generate probabilities of the input layer
 - incuding the biased probability
 -
 - transpose of the hiddenProbs function
 -
 - map sigmoid $ (transpose inputs) `mmult` weights
 - 
 --} 
{-# INLINE inputProbs #-}
inputProbs :: RBM -> Array U DIM1 Double -> Array U DIM1 Double
inputProbs rb hidden = R.computeUnboxedS $ R.fromFunction shape (computeInputProb wws hidden)
   where
      shape = (Z :. (numInputs rb))
      wws = weights rb

{--
 - sigmoid of the dot product of the col
 --}
{-# INLINE computeInputProb #-}
computeInputProb :: Array U DIM2 Double -> Array U DIM1 Double -> DIM1 -> Double
computeInputProb wws hidden xi = sigmoid sm
   where
      rw = R.slice wws (Any :. (pos xi))
      sm = R.sumAllS $ R.zipWith (*) rw hidden

-- -- update the rbm weights from each batch
-- learn :: RandomGen r => r -> RBM -> [[[Double]]] -> RBM
-- learn _ rb [] = rb
-- learn rand rb iis = nrb `deepseq` learn r2 nrb (tail iis)
--    where
--       (r1,r2) = split rand
--       nrb = batch r1 rb (head iis)
-- 
-- -- given a batch of unbiased inputs, update the rbm weights from the batch at once 
-- batch :: RandomGen r => r -> RBM -> [[Double]] -> RBM
-- batch rand rb inputs = rb { weights = uw }
--    where
--       uw = zipWith (+) (weights rb) wd
--       wd = weightUpdatesLoop rand rb inputs (take len [0..])
--       len = ((numInputs rb) + 1) * ((numHidden rb) + 1)
-- 
-- -- given a batch of unbiased inputs, generate the the RBM weight updates for the batch
-- weightUpdatesLoop :: RandomGen r => r -> RBM -> [[Double]] -> [Double] -> [Double]
-- weightUpdatesLoop _ _ [] pw = pw
-- weightUpdatesLoop rand rb (inputs:rest) pw = weightUpdatesLoop r2 rb rest npw
--    where
--       npw = pw `deepseq` zipWith (+) pw wd
--       wd = weightUpdate r1 rb inputs
--       (r1,r2) = split rand
-- 
-- -- given an unbiased input, generate the the RBM weight updates
-- weightUpdate :: RandomGen r => r -> RBM -> [Double] -> [Double]
-- weightUpdate rand rb inputs = zipWith (-) w1 w2
--    where
--       (r1,r2) = split rand
--       hiddens = generate r1 rb (1:inputs)
--       newins = regenerate r2 rb hiddens
--       w1 = vmult hiddens (1:inputs)
--       w2 = vmult hiddens newins 
-- 
-- -- given a biased input (1:input), generate a biased hidden layer sample
-- generate :: RandomGen r => r -> RBM -> [Double] -> [Double]
-- generate rand rb inputs = zipWith applyP (hiddenProbs rb inputs) (0:(randomRs (0,1) rand))
-- 
-- -- given a biased hidden layer sample, generate a biased input layer sample
-- regenerate :: RandomGen r => r -> RBM -> [Double] -> [Double]
-- regenerate rand rb hidden = zipWith applyP (inputProbs rb hidden) (0:(randomRs (0,1) rand))
-- 
-- 
-- 
-- --sample is 0 if generated number gg is greater then probabiliy pp
-- --so the higher pp the more likely that it will generate a 1
-- applyP :: Double -> Double -> Double
-- applyP pp gg | pp < gg = 0
--              | otherwise = 1
-- 
-- -- row vec * col vec
-- -- or (m x 1) * (1 x c) matrix multiply 
-- vmult :: [Double] -> [Double] -> [Double]
-- vmult xxs yys = [ (xx*yy) | xx <- xxs, yy<-yys]
-- 

-- sigmoid function
{-# INLINE sigmoid #-}
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

-- 
-- -- tests
-- 
-- -- test to see if we can learn a random string
-- prop_learned :: Word8 -> Word8 -> Bool
-- prop_learned ni' nh' = (tail regened) == input
--    where
--       regened = regenerate (mr 2) lrb $ generate (mr 3) lrb (1:input)
--       --learn the inputs
--       lrb = learn (mr 1) rb [inputs]
--       rb = rbm (mr 0) ni nh
--       inputs = replicate 2000 $ input
--       --convert a random list of its 0 to 1 to doubles
--       input = map fromIntegral $ take ni $ randomRs (0::Int,1::Int) (mr 4)
--       ni = fromIntegral ni' :: Int
--       nh = fromIntegral nh' :: Int
--       --creates a random number generator with a seed
--       mr i = mkStdGen (ni + nh + i)
-- 
-- 
-- prop_learn :: Word8 -> Word8 -> Bool
-- prop_learn ni nh = ln == (length $ weights $ lrb)
--    where
--       ln = ((fi ni) + 1) * ((fi nh) + 1)
--       lrb = learn' rb 1
--       learn' rr ix = learn (mkStdGen ix) rr [[(take (fi ni) $ cycle [0,1])]]
--       rb = rbm (mkStdGen 0) (fi ni) (fi nh)
--       fi = fromIntegral
-- 
-- prop_batch :: Word8 -> Word8 -> Word8 -> Bool
-- prop_batch ix ni nh = ln == (length $ weights $ lrb)
--    where
--       ln = ((fi ni) + 1) * ((fi nh) + 1)
--       lrb = batch rand rb inputs
--       rb = rbm rand (fi ni) (fi nh)
--       rand = mkStdGen ln
--       inputs = replicate (fi ix) $ take (fi ni) $ cycle [0,1]
--       fi = fromIntegral

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = (fi ni) * (fi nh)  == (length $ R.toList $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi :: Word8 -> Int
      fi ww = 1 + (fromIntegral ww)

-- prop_vmult :: Bool
-- prop_vmult = vmult [1,2,3] [4,5] == [1*4,1*5,2*4,2*5,3*4,3*5]
-- 
-- prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
-- prop_hiddenProbs gen ni nh = (fi nh) + 1 == length pp
--    where
--       pp = hiddenProbs rb $ replicate ((fi ni) + 1) 0.0
--       rb = rbm (mkStdGen gen) (fi ni) (fi nh)
--       fi = fromIntegral
-- 
-- prop_hiddenProbs2 :: Bool
-- prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
--    where
--       h0 = w00 * i0 + w01 * i1 + w02 * i2  
--       h1 = w10 * i0 + w11 * i1 + w12 * i2 
--       i0:i1:i2:_ = [1..]
--       w00:w01:w02:w10:w11:w12:_ = [1..]
--       wws = [w00,w01,w02,w10,w11,w12]
--       pp = hiddenProbs rb [i0,i1,i2]
--       rb = RBM wws 2 1
-- 
-- prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
-- prop_inputProbs gen ni nh = (fi ni) + 1 == length pp
--    where
--       pp = inputProbs rb $ replicate ((fi nh) + 1) 0.0
--       rb = rbm (mkStdGen gen) (fi ni) (fi nh)
--       fi = fromIntegral
-- 
-- prop_inputProbs2 :: Bool
-- prop_inputProbs2 = pp == map sigmoid [i0,i1,i2]
--    where
--       i0 = w00 * h0 + w10 * h1
--       i1 = w01 * h0 + w11 * h1
--       i2 = w02 * h0 + w12 * h1
--       h0:h1:_ = [1..]
--       w00:w01:w02:w10:w11:w12:_ = [1..]
--       wws = [w00,w01,w02,w10,w11,w12]
--       pp = inputProbs rb [h0,h1]
--       rb = RBM wws 2 1
-- 
prop_energy :: Int -> Int -> Int -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb input
      input = randomishDoubleArray (Z :. (fi ni)) 0 1 gen
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ww = 1 + (fromIntegral ww)

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "init"     prop_init
   runtest "energy"   prop_energy
--    runtest "hiddenp"  prop_hiddenProbs
--    runtest "hiddenp2" prop_hiddenProbs2
--    runtest "inputp"   prop_inputProbs
--    runtest "inputp2"  prop_inputProbs2
--    runtest "vmult"    prop_vmult
--    runtest "learn"    prop_learn
--    runtest "batch"    prop_batch
--    runtest "learned"  prop_learned

perf :: IO ()
perf = do
   let file = "dist/perf-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 1.0 }
       sz = 4096
       input = randomishDoubleArray (Z :. sz) 0 1 0
       rb = rbm (mkStdGen 0) sz sz
   defaultMainWith cfg [
         bgroup "energy" [ bench (show sz) $ whnf (energy rb) input
                         ]
      ]
--       ,bgroup "hidden" [ bench "3x3"  $ whnf (prop_hiddenProbs 0 3) 3
--                        , bench "63x63"  $ whnf (prop_hiddenProbs 0 63) 63
--                        , bench "127x127"  $ whnf (prop_hiddenProbs 0 127) 127
--                        ]
--       ,bgroup "input" [ bench "3x3"  $ whnf (prop_inputProbs 0 3) 3
--                       , bench "63x63"  $ whnf (prop_inputProbs 0 63) 63
--                       , bench "127x127"  $ whnf (prop_inputProbs 0 127) 127
--                       ]
--       ,bgroup "batch" [ bench "3"  $ whnf (prop_batch 3 63) 63
--                       , bench "63"  $ whnf (prop_batch 63 63) 63
--                       , bench "127"  $ whnf (prop_batch 127 63) 63
--                       ]
--       ]
--    putStrLn $ "perf log written to " ++ file
