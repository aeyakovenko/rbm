module RBM(rbm
          ,sample
          ,test
          ,energy
          ) where
import System.Exit (exitFailure)

import Control.Monad.State.Lazy(runState
                               ,get
                               ,State
                               ,put
                               )
import System.Random(RandomGen
                    ,randomRs
                    ,split
                    )
import Control.Applicative((<$>))

data RBM = RBM { weights :: [Double]
               , numInputs :: Int
               , numHidden :: Int
               }

rbm :: Int -> Int -> RBM
rbm ni nh = RBM [] ni nh

energy :: RBM -> [Double] -> Double
energy rb inputs = negate ee
   where
      ni = numInputs rb
      nh = numHidden rb
      hhs = hiddenProbs rb inputs
      wws = weights rb
      ee = sum [(inputs !! ii) * (hhs !! jj) * (wws !! (ii * nh + jj)) | ii <- [0..ni], jj <- [0..nh]]

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

hiddenProbs :: RBM -> [Double] -> [Double]
hiddenProbs rb inputs = [sigmoid (sums (weights rb) jj) | jj <- [0..nh]] 
   where
      nh = numHidden rb
      ni = numInputs rb
      sums wws jj = sum [ (wws !! (ii * nh + jj)) * (inputs !! ii) | ii <- [0..ni] ]

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

test :: IO ()
test = do
   exitFailure
