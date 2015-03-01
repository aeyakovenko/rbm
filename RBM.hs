module RBM(rbm
          ,sample
          ,energy
          ,test
          ,bench
          ) where

import Criterion.Main(defaultMain,bgroup,bench,whnf)
import Data.Word(Word8)
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckResult)
import Test.QuickCheck.Test(isSuccess)
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

data RBM = RBM { weights :: [Double] -- input numHidden x numInputs 
               , numInputs :: Int
               , numHidden :: Int
               }

rbm :: Int -> Int -> RBM
rbm ni nh = RBM [] ni nh

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
hiddenProbs rb biased = map (sigmoid . sum) $ groupByN (ni + 1) $ zipWith (*) wws $ concat $ repeat biased 
   where
      wws = weights rb
      ni = numInputs rb

inputProbs :: RBM -> [Double] -> [Double]
inputProbs rb hidden = [ sigmoid (sums (weights rb) ii) | ii <- [0..ni] ] 
   where
      ni = numInputs rb
      nh = numHidden rb
      sums wws ii = sum [ (wws !! (ii * nh + jj)) * (hidden !! ii) | jj <- [0..nh] ]

initWeights :: RandomGen r => RBM -> State r RBM
initWeights (RBM [] ni nh) = do
   nw <- take ((nh + 1)* (ni + 1)) <$> randomRs (0,1) <$> getR
   return $ RBM nw ni nh
initWeights rb = return $ rb   

sample :: RandomGen r => r -> RBM -> [Double] -> RBM
sample rand rb' inputs = fst $ (flip runState) rand $ do
   rb <- initWeights rb'
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
vmult inputs hidden = concat [ line xx | xx <- inputs]
   where
      line xx = [ (xx * yy) | yy <- hidden] 

getR :: RandomGen r => State r r
getR = do
   (rd,rd') <- split <$> get
   put rd'
   return $ rd

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = ((fi ni) + 1) * ((fi nh) + 1) == (length $ weights $ fst $ runst $ doinit)
   where
      runst = (flip runState) (mkStdGen gen)
      doinit = initWeights $ rbm (fi ni) (fi nh)
      fi = fromIntegral

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb $ replicate (fi ni) 0.0
      rb = fst $ runst $ doinit
      runst = (flip runState) (mkStdGen gen)
      doinit = initWeights $ rbm (fi ni) (fi nh)
      fi = fromIntegral

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       runtest p = check =<< verboseCheckResult p 
   runtest prop_init
   runtest prop_energy

bench :: IO ()
bench = do
bench = defaultMain [ 
   bgroup "energy" [ bench "127x127"  $ whnf prop_energy 0 127 127
                   , bench "255x255"  $ whnf prop_energy 0 255 255
                   ]
   ]



   
