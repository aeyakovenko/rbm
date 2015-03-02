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

--impl modules
import Control.DeepSeq(NFData,rnf,deepseq)
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

data RBM = RBM { weights :: [Double] -- weight matrix with 1 bias nodes, numHidden + 1 x numInputs  + 1
               , numInputs :: Int    -- size without bias node
               , numHidden :: Int    -- size without bias node
               }

instance NFData RBM where
   rnf (RBM wws ni nh) = rnf (wws,ni,nh)

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
      ee = wws `deepseq` hhs `deepseq` sum $ zipWith (*) wws [inp * hid | inp <- biased, hid <- hhs]

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

groupByN :: Int -> [a] -> [[a]]
groupByN _ [] = []
groupByN n ls = (take n ls) : groupByN n (drop n ls)

{--
    w0 w1 w2
w0  00 01 02     i0     h0 = w00 * i0 + w01 * i1 + w02 * i2  
w1  10 11 12  x  i1     h1 = w10 * i0 + w11 * i1 + w12 * i2  
                 i2
--}

hiddenProbs :: RBM -> [Double] -> [Double]
hiddenProbs rb biased = map (sigmoid . sum) $ groupByN ni $ zipWith (*) wws $ cycle biased
   where
      wws = weights rb
      ni = (numInputs rb) + 1

{--
    w0 w1 w2
w0  00 01 02 
w1  10 11 12 
       x
    h0 h1

i0 = w00 * h0 + w10 * h1
i1 = w01 * h0 + w11 * h1
i2 = w02 * h0 + w12 * h1

 --}
inputProbs :: RBM -> [Double] -> [Double]
inputProbs rb hidden = map (sigmoid . sum) $ transpose $ groupByN ni $ zipWith (*) wws hhs
   where
      hhs = concat $ transpose $ replicate ni hidden
      wws = weights rb
      ni = (numInputs rb) + 1
      nh = (numHidden rb) + 1

sample :: RandomGen r => r -> RBM -> [Double] -> RBM
sample rand rb inputs = fst $ (flip runState) rand $ do
   let nh = numHidden rb
   hrands <- (:) 0 <$> take nh <$> randomRs (0,1) <$> getR
   let biased = 1:inputs
       hprobs = hiddenProbs rb biased
       ni = numInputs rb
       applyp pp gg | pp > gg = 1
                    | otherwise = 0
       hiddenSample = zipWith applyp hprobs hrands 
       w1 = vmult hiddenSample biased 
       iprobs = inputProbs rb hiddenSample
   irands <- (:) 0 <$> take ni <$>  randomRs (0,1) <$> getR
   let inputSample = zipWith applyp iprobs irands 
       w2 = vmult hiddenSample inputSample 
       wd = zipWith (-) w1 w2
       uw = zipWith (+) (weights rb) wd
   return $ rb { weights = uw }

batch :: RandomGen r => r -> RBM -> [[Double]] -> RBM
batch _ rb [] = rb
batch rand rb iis =
   let (rr,nr) = split rand
       nrb = sample rr rb (head iis) 
   in  nrb `deepseq` batch nr nrb (tail iis)

-- row vec * col vec
vmult :: [Double] -> [Double] -> [Double]
vmult xxs yys = [ (xx*yy) | xx <- xxs, yy<-yys]

getR :: RandomGen r => State r r
getR = do
   (rd,rd') <- split <$> get
   put rd'
   return $ rd

prop_sample :: Word8 -> Word8 -> Bool
prop_sample ni nh = ln == (length $ weights $ lrb)
   where
      ln = ((fi ni) + 1) * ((fi nh) + 1)
      lrb = learn rb 1
      learn rr ix = sample (mkStdGen ix) rr (take (fi ni) $ cycle [0,1])
      rb = rbm (mkStdGen 0) (fi ni) (fi nh)
      fi = fromIntegral

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = ln == (length $ weights $ lrb)
   where
      ln = ((fi ni) + 1) * ((fi nh) + 1)
      lrb = batch rand rb inputs
      rb = rbm rand (fi ni) (fi nh)
      rand = mkStdGen ln
      inputs = replicate (fi ix) $ take (fi ni) $ cycle [0,1]
      fi = fromIntegral

test_sample = map toSample $ tail $ generatedInputs
   where
      input = [0,1]
      generatedInputs = inputProbs lrb generatedHiddenSample 
      generatedHiddenSample = setBias $ map toSample $ hiddenProbs lrb (1:input)
      setBias ls = 1:(tail ls)
      toSample pp | pp >  0.9 = 1
                  | otherwise = 0
      ni = 2
      nh = 2
      inputs = replicate 1000 $ take ni $ cycle input
      rand = mkStdGen 0
      lrb = batch rand rb inputs
      rb = rbm (mkStdGen 0) ni nh

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = ((fi ni) + 1) * ((fi nh) + 1) == (length $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_vmult :: Bool
prop_vmult = vmult [1,2,3] [4,5] == [1*4,1*5,2*4,2*5,3*4,3*5]

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = (fi nh) + 1 == length pp
   where
      pp = hiddenProbs rb $ replicate ((fi ni) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
   where
      h0 = w00 * i0 + w01 * i1 + w02 * i2  
      h1 = w10 * i0 + w11 * i1 + w12 * i2 
      i0:i1:i2:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      pp = hiddenProbs rb [i0,i1,i2]
      rb = RBM wws 2 1

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = (fi ni) + 1 == length pp
   where
      pp = inputProbs rb $ replicate ((fi nh) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_inputProbs2 :: Bool
prop_inputProbs2 = pp == map sigmoid [i0,i1,i2]
   where
      i0 = w00 * h0 + w10 * h1
      i1 = w01 * h0 + w11 * h1
      i2 = w02 * h0 + w12 * h1
      h0:h1:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      pp = inputProbs rb [h0,h1]
      rb = RBM wws 2 1

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
   runtest prop_hiddenProbs2
   runtest prop_inputProbs
   runtest prop_inputProbs2
   runtest prop_vmult
   runtest prop_sample
   runtest prop_batch

perf :: IO ()
perf = do
   let file = "dist/perf-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 0.5 }
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
      ,bgroup "batch" [ bench "15"  $ whnf (prop_batch 15 63) 63
                      , bench "63"  $ whnf (prop_batch 63 63) 63
                      , bench "127"  $ whnf (prop_batch 127 63) 63
                      , bench "255"  $ whnf (prop_batch 255 63) 63
                      ]
      ]
   putStrLn $ "perf log written to " ++ file
