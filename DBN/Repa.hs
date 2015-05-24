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
               ,BxH(BxH)
               ,HxB(unHxB)
               ,IxH(IxH)
               ,RBM
               ,rbm
               )
import qualified Data.Array.Repa as R
import Data.Array.Repa(Z(Z)
                      ,(:.)((:.))
                      )
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import qualified Data.Array.Repa.Algorithms.Pixel as R
import System.Random(RandomGen
                    ,split
                    ,mkStdGen
                    ,randomRs
                    ,random
                    ,newStdGen
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
   let loop !db' (pp,ix) = do
          let tdb = take ix db'
              rdb = drop ix db'
          ndb <- learnLast batches pp tdb
          return $ ndb ++ rdb
   foldM loop db $ zip params [1..]

learnLast :: (Monad m)  => [m BxI] -> RBM.Params -> DBN -> m DBN
learnLast _ _ [] = return []
learnLast batches pars [!rb] = do 
   !nrb <- RBM.learn pars rb batches
   return $ nrb : []
learnLast batches pars (!nrb:rest) = do
   let (r1:r2:_) = splits $ mkStdGen (RBM.seed pars)
       npars = pars { RBM.seed = fst $ random r1 }
       gen rb mbxi = do
         bxi <- mbxi
         !hxb <- bxi `deepseq` RBM.generate r2 rb bxi
         BxI <$> (R.transpose2P $ unHxB hxb)
   let nbs = map (gen nrb) batches
   nrbms <- learnLast nbs npars rest
   return $ nrb : nrbms

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

printImages:: String -> DBN -> IO ()
printImages sname db = do
   let imagewidth = 28
       computeStrip (BxI bxi) (Z :. rix :. cix) = 
         let  numimages = R.row $ R.extent $ bxi
              imagenum = cix `div` numimages
              imagepixel = rix * (imagewidth) + (cix `mod` numimages)
         in   R.rgb8OfGreyDouble $ bxi R.! ( Z :. imagenum :. (imagepixel + 1))
       regenSample ix = do 
            let sfile = concat [sname, (show ix), ".bmp"]
            putStrLn $ concat ["generatint strip: ", sfile]
            let name = "dist/sample" ++ (show ix)
                readBatch = BxI <$> (readArray name)
            g1 <- newStdGen
            bxi <- readBatch
            bxh <- generate g1 db bxi
            g2 <- newStdGen
            bxi' <- regenerate g2 db bxh
            let rows = R.row $ R.extent $ unBxI bxi'
            let sh = Z :. imagewidth :. (imagewidth * rows)
            strip <- R.computeUnboxedP $ R.fromFunction sh (computeStrip bxi')
            R.writeImageToBMP sfile strip
   mapM_ regenSample [0..9::Int] 

mnist :: IO ()
mnist = do 
   gen <- newStdGen
   let [r0,r1,r2] = dbn gen [785,501,501,11]
       name ix = "dist/train" ++ (show ix)
       readBatch ix = BxI <$> (readArray (name ix))
       iobatches = map readBatch [0..468::Int]
       p1 = RBM.params { RBM.rate = 0.01, RBM.minMSE = 0.1 }
       p2 = RBM.params { RBM.rate = 0.001, RBM.minMSE = 0.01 }
       
   d1 <- learnLast iobatches p1 [r0]
   printImages "dist/strip1." d1
   d2 <- learnLast iobatches p2 d1
   printImages "dist/strip2." d2
   d3 <- learnLast iobatches p1 (d2 ++ [r1])
   printImages "dist/strip3." d3
   d4 <- learnLast iobatches p2 d3
   printImages "dist/strip4." d4
   d5 <- learnLast iobatches p1 (d4 ++ [r2])
   printImages "dist/strip5." d5
   d6 <- learnLast iobatches p2 d5
   printImages "dist/strip6." d6

