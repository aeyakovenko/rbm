module RBM(rbm
          ,sample
          ,energy
          ,test
          ,perf
          ) where

--benchmark modules
import Criterion.Main(defaultMainWith,defaultConfig,bgroup,bench,whnf)
import Criterion.Types(reportFile,timeLimit)
--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess)

import Data.Word(Word8)
import Data.List(transpose)
import Control.Monad.State.Lazy(runState
                               ,get
                               ,State
                               ,put
                               )
import System.Random(RandomGen
                    ,randomRs
                    ,split
                    ,mkStdGen
                    )
import Control.Applicative((<$>))

data RBM = RBM { weights :: [Double] -- weight matrix, numHidden + 1 x numInputs  + 1
               , numInputs :: Int
               , numHidden :: Int
               }

rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = RBM nw ni nh
   where
      nw = take ((nh + 1)* (ni + 1)) $ randomRs (0,1) r

energy :: RBM -> [Double] -> Double
energy rb inputs = negate ee
   where
      biased = 1:inputs
      hhs = hiddenProbs rb biased
      wws = weights rb
      ee = sum $ zipWith (*) wws [inp * hid | inp <- biased, hid <- hhs]

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

groupByN :: Int -> [a] -> [[a]]
groupByN _ [] = []
groupByN n ls = (take n ls) : groupByN n (drop n ls)

hiddenProbs :: RBM -> [Double] -> [Double]
hiddenProbs rb biased = map (sigmoid . sum) $ groupByN ni $ zipWith (*) wws $ cycle biased 
   where
      wws = weights rb
      ni = (numInputs rb) + 1

inputProbs :: RBM -> [Double] -> [Double]
inputProbs rb hidden = map (sigmoid . sum) $ groupByN nh $ zipWith (*) wws hhs
   where
      hhs = concat $ transpose $ replicate ni hidden
      wws = weights rb
      ni = (numInputs rb) + 1
      nh = (numHidden rb) + 1

sample :: RandomGen r => r -> RBM -> [Double] -> RBM
sample rand rb inputs = fst $ (flip runState) rand $ do
   let nh = numHidden rb
   hrands <- take nh <$> (:) 1 <$> randomRs (0,1) <$> getR
   let biased = 1:inputs
       hprobs = hiddenProbs rb biased
       ni = numInputs rb
       applyp pp gg | pp < gg = 1
                    | otherwise = 0
       hiddenSample = zipWith applyp hprobs hrands 
       w1 = vmult biased hiddenSample
       iprobs = inputProbs rb hiddenSample
   irands <- take ni <$> (:) 1 <$> randomRs (0,1) <$> getR
   let inputSample = zipWith applyp iprobs irands 
       w2 = vmult inputSample hiddenSample
       wd = zipWith (-) w1 w2
       uw = zipWith (+) (weights rb) wd
   return $ rb { weights = uw }

vmult :: [Double] -> [Double] -> [Double]
vmult inputs hidden = [ (xx*yy) | xx <- inputs, yy<-hidden]

getR :: RandomGen r => State r r
getR = do
   (rd,rd') <- split <$> get
   put rd'
   return $ rd

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = ((fi ni) + 1) * ((fi nh) + 1) == (length $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = (fi nh) + 1 == length pp
   where
      pp = hiddenProbs rb $ replicate ((fi ni) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = (fi ni) + 1 == length pp
   where
      pp = inputProbs rb $ replicate ((fi nh) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb $ replicate (fi ni) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 50 } 
       runtest p = check =<< verboseCheckWithResult cfg p 
   runtest prop_init
   runtest prop_energy
   runtest prop_hiddenProbs
   runtest prop_inputProbs

perf :: IO ()
perf = do
   let cfg = defaultConfig { reportFile = Just "dist/perf-RBM.html", timeLimit = 0.5 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "3x3"  $ whnf (prop_energy 0 3) 3
                       , bench "127x127"  $ whnf (prop_energy 0 127) 127
                       , bench "255x255"  $ whnf (prop_energy 0 255) 255
                       ]
      ,bgroup "hidden" [ bench "3x3"  $ whnf (prop_hiddenProbs 0 3) 3
                       , bench "127x127"  $ whnf (prop_hiddenProbs 0 127) 127
                       , bench "255x255"  $ whnf (prop_hiddenProbs 0 255) 255
                       ]
      ,bgroup "input" [ bench "3x3"  $ whnf (prop_inputProbs 0 3) 3
                      , bench "127x127"  $ whnf (prop_inputProbs 0 127) 127
                      , bench "255x255"  $ whnf (prop_inputProbs 0 255) 255
                      ]
      ]
   putStrLn "perf log written to dist/perf-RBM.html"
