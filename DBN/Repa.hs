{-# LANGUAGE BangPatterns #-}
module DBN.Repa(dbn
               ,learn
               ,generate
               ,perf
               ,test
               ,mnist
               ) where

import qualified RBM.Repa as RBM
import RBM.Repa(BxI(BxI,unBxI)
               ,BxH(BxH,unBxH)
               ,HxB(HxB,unHxB)
               ,IxH(IxH)
               ,RBM
               ,rbm
               )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import System.Random(RandomGen
                    ,split
                    ,mkStdGen
                    ,randomRs
                    )
import Control.DeepSeq(deepseq)

import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
import Data.Mnist(readArray)
import Control.Monad(foldM)
import Control.Monad.Identity(runIdentity)


type DBN = [RBM]
type PV = R.Array R.U R.DIM1 Double

{--
 - create a dbn with some randomized weights
 --}
dbn :: RandomGen r => r -> [Int] -> DBN
dbn _ [] = error "dbn: empty size list" 
dbn _ (_:[]) = error "dbn: needs more then 1 size" 
dbn rand (ni:nh:[]) = [rbm rand ni nh]
dbn rand (ni:nh:rest) = rbm r1 ni nh : dbn r2 (nh:rest)
   where
      (r1,r2) = split rand

{--
 - teach the dbn a batch of inputs
 --}
learn :: (Monad m)  => [RBM.Params] -> DBN -> [m BxI] -> m DBN
learn params db batches = do
   foldM (\ db' ix -> learnLayer ix params db' batches) db [0..(length db) - 1]

learnLayer :: (Monad m)  => Int -> [RBM.Params] -> DBN -> [m BxI] -> m DBN
learnLayer _ _ [] _ = return []
learnLayer 0 (pars:_) (!rb:rest) batches = do 
   !nrb <- RBM.learn pars rb batches
   return $ nrb : rest
learnLayer lvl (pars:npars) (!nrb:rest) batches = do
   let rr = mkStdGen (RBM.seed pars)
       gen rb mbxi = do
         bxi <- mbxi
         !hxb <- bxi `deepseq` RBM.generate rr rb bxi
         BxI <$> (R.transpose2P $ unHxB hxb)
   let nbs = map (gen nrb) batches
   nrbms <- learnLayer (lvl - 1) npars rest nbs
   return $ nrb : nrbms
learnLayer _ [] _ _ = error "dbn: not enough learning parameters"

{--
 - generate a batch of output
 --}
generate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxI -> m BxH
generate _ [] pb = return $ RBM.BxH $ RBM.unBxI pb

generate rand (rb:rest) pb = do 
   let (r1:rn:_) = splits rand
   hxb <- RBM.unHxB <$> RBM.generate r1 rb pb
   bxi <- BxI <$> (R.transpose2P hxb)
   generate rn rest bxi
{-# INLINE generate #-}

{--
 - regenerate a batch of input
 --}
regenerate :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH -> m BxI
regenerate r dbn' bxh = regenerate' r (reverse dbn') bxh
{-# INLINE regenerate #-}

regenerate' :: (Functor m, Monad m, RandomGen r) => r -> DBN -> BxH ->  m BxI
regenerate' _ [] pb = return $ RBM.BxI $ RBM.unBxH pb

regenerate' rand (rb:rest) pb = do 
   let (r1:rn:_) = splits rand
   ixh <- IxH <$> R.transpose2P (RBM.unHxI rb)
   ixb <- RBM.unIxB <$> RBM.regenerate r1 ixh pb
   bxh <- BxH <$> (R.transpose2P ixb)
   regenerate' rn rest bxh
{-# INLINE regenerate' #-}

{--
 - given a batch of generated hidden nodes
 - calculate the probability of each node being one
 - basically, a way to sample the probability if the batch 
 - from the same class of inputs
 --}
probV :: Monad m => HxB -> m PV 
probV hxb = do 
   sums <- R.sumP (RBM.unHxB hxb)
   let nb = R.col $ R.extent $ RBM.unHxB hxb
   R.computeUnboxedP $ R.map (prob nb) sums

prob :: Int -> Double -> Double
prob nb xx = xx / (fromIntegral nb)

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> Int -> ([Double],[Double])
run_prop_learned rate ni nd nh = runIdentity $ do
   let rb = dbn (mr 0) layers
       nmin = (fi ni) `min` (fi nh)
       nmax = (fi ni) `max` (fi nh)
       layers = (fi ni) : (take (fi nd) ((randomRs (nmin,nmax)) (mr 5))) ++ [nh]
       inputbatchL = concat $ replicate batchsz inputlst
       inputbatch = BxI $ R.fromListUnboxed (R.Z R.:. batchsz R.:. fi ni) $ inputbatchL
       inputarr = BxI $ R.fromListUnboxed (R.Z R.:. 1 R.:. fi ni) $ inputlst
       geninputs = randomRs (0::Int,1::Int) (mr 4)
       inputlst = map fromIntegral $ take (fi ni) $ 1:geninputs
       fi ww = 1 + ww
       mr i = mkStdGen (fi ni + fi nh + i)
       batchsz = 2000
       par = RBM.params { RBM.rate = rate }
   lbn <- learn (take (length rb) $ repeat par) rb [return inputbatch]
   bxh <- generate (mr 3) lbn inputarr
   bxi <- regenerate (mr 2) lbn bxh
   return $ ((tail $ R.toList $ unBxI $ bxi), (tail $ R.toList $ unBxI $ inputarr))

prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = (uncurry (==)) $ run_prop_learned 1.0 (fi ni) 2 (fi nh)
   where
      fi = fromIntegral

prop_not_learned :: Word8 -> Word8 -> Bool
prop_not_learned ni nh = (uncurry check) $ run_prop_learned (-1.0) (fi ni) 2 (fi nh)
   where
      fi ii = fromIntegral ii
      check aas bbs = (null aas) || (aas /= bbs) || (minimum aas == 1.0)
 
test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "notlearnred"  prop_not_learned
   runtest "learned"      prop_learned

perf :: IO ()
perf = return ()

printImages:: DBN -> IO ()
printImages dbn = do
   let imagewidth = 28
       computeStrip (BxI bxi) (Z :. rix :. cix) = 
         let  numimages = row $ R.extent $ bxi
              imagenum = cix `div` numimages
              imagepixel = rix * (imagewidth) + (cix `mod` numimages)
         in   bxi ! ( Z :. imagenum :. (imagepixel + 1))
       regenSample ix = do 
            let name ix = "dist/sample" ++ (show ix)
                readBatch ix = BxI <$> (readArray (name ix))
            g1 <- newStdGen
            bxi <- readBatch ix
            bxh <- generate g1 dbn bxi
            g2 <- newStdGen
            bxi <- regenerate g2 dbn bxh
            let rows = row $ R.extent $ unBxI bxi
            let sh = Z :. imagewidth :. (imagewidth * rows)
            strip <- d2u $ fromFunction sh (comptueStrip bxi)
            R.writeMatrixToGreyscaleBMP ("dist/strip" ++ (show ix) ++ ".bmp") strip
      mapM_ regenSample [0..9] 

mnist :: IO ()
mnist = do 
   let gen = mkStdGen 0
       d0 = dbn gen [785,501,501,11]
       name ix = "dist/train" ++ (show ix)
       readBatch ix = BxI <$> (readArray (name ix))
       iobatches = map readBatch [0..468::Int]
       p1 = RBM.params { RBM.rate = 0.01, RBM.minMSE = 0.1 }
       p2 = RBM.params { RBM.rate = 0.001, RBM.minMSE = 0.01 }
       
   d1 <- learnLayer 0 [p1] d0 iobatches
   d2 <- learnLayer 0 [p2] d1 iobatches
   d3 <- learnLayer 1 [p2,p1] d2 iobatches
   d4 <- learnLayer 1 [p2,p2] d3 iobatches
   d5 <- learnLayer 2 [p2,p2,p1] d4 iobatches
   d6 <- learnLayer 2 [p2,p2,p2] d5 iobatches
   let
      gen = mkStdGen 0
      pars = [RBM.params { RBM.rate = 0.01 },RBM.params { RBM.rate = 0.01 },RBM.params { RBM.rate = 0.01 }]
      trainBatch :: DBN -> Int -> IO DBN
      trainBatch db lvl = do
         let name ix = "dist/train" ++ (show ix)
             readBatch ix = BxI <$> (readArray (name ix))
             iobatches = map readBatch $ take 20 (randomRs (0::Int,468) gen)
         putStrLn $ concat ["training: layer: ", (show lvl)]
         learnLayer lvl pars db iobatches
      testBatch :: DBN -> Int -> IO ()
      testBatch db ix = do
         let name = "dist/test" ++ (show ix)
         b <- readArray name
         bxh <- generate gen db $ BxI b
         hxb <- HxB <$> (R.transpose2P $ unBxH bxh)
         pv <- probV hxb
         print (ix, R.toList pv)

   de <- foldM trainBatch ds [0..(length pars)]
   mapM_ (testBatch de) [0..9] 
