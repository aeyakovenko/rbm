--from https://github.com/mhwombat/backprop-example/blob/master/Mnist.hs
{-# LANGUAGE FlexibleInstances #-}
module Data.Mnist (generateTrainBatches
                  ,generateTestBatches
                  ,readArray
                  ,generateBigTrainBatches
                  ,generateSamples
                  ,mnist
                  )
  where

import qualified Data.ByteString.Lazy as BL
import Data.Binary.Get
import qualified Data.Binary as B
import Data.Word
import qualified Data.List.Split as S
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import qualified Data.Array.Repa.Algorithms.Pixel as R
import Data.Array.Repa(Z(Z)
                      ,(:.)((:.))
                      )
import Codec.Compression.GZip as GZ
import Data.List.Split(chunksOf)
import System.Random(newStdGen, randomRs)

import qualified DBN.Repa as DBN
import qualified RBM.Repa as RBM


data Image = Image {
      iRows :: Int
    , iColumns :: Int
    , iPixels :: [Word8]
    } deriving (Eq, Show)

toMatrix :: [Image] -> R.Array R.U R.DIM2 Double
toMatrix images = m
  where 
        m = R.fromListUnboxed (R.Z R.:. len R.:. maxsz) (concatMap pixels images)
        maxsz = 1 + (maximum $ map (\ ii -> (iRows ii) * (iColumns ii)) images)
        len = length images
        pixels im = take maxsz $ 1:((normalisedData im) ++ [0..])

{-
toColumnVector :: Image -> Matrix Double
toColumnVector i = (r><1) q :: Matrix Double
  where r = Mnist.rows i * Mnist.columns i
        p = map fromIntegral (pixels i)
        q = map normalise p
-}

normalisedData :: Image -> [Double]
normalisedData image = map normalisePixel (iPixels image)

--normalisedData :: Image -> [Double]
--normalisedData i = map (/m) x 
--    where x = map normalisePixel (pixels i)
--          m = sqrt( sum (zipWith (*) x x))

normalisePixel :: Word8 -> Double
normalisePixel p = (fromIntegral p) / 255.0

-- MNIST label file format
--
-- [offset] [type]          [value]          [description]
-- 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
-- 0004     32 bit integer  10000            number of items
-- 0008     unsigned byte   ??               label
-- 0009     unsigned byte   ??               label
-- ........
-- xxxx     unsigned byte   ??               label
--
-- The labels values are 0 to 9.

deserialiseLabels :: Get (Word32, Word32, [Word8])
deserialiseLabels = do
  magicNumber <- getWord32be
  count <- getWord32be
  labelData <- getRemainingLazyByteString
  let labels = BL.unpack labelData
  return (magicNumber, count, labels)

readLabels :: FilePath -> IO [Int]
readLabels filename = do
  content <- GZ.decompress <$> BL.readFile filename
  let (_, _, labels) = runGet deserialiseLabels content
  return (map fromIntegral labels)


-- MNIST Image file format
--
-- [offset] [type]          [value]          [description]
-- 0000     32 bit integer  0x00000803(2051) magic number
-- 0004     32 bit integer  ??               number of images
-- 0008     32 bit integer  28               number of rows
-- 0012     32 bit integer  28               number of columns
-- 0016     unsigned byte   ??               pixel
-- 0017     unsigned byte   ??               pixel
-- ........
-- xxxx     unsigned byte   ??               pixel
-- 
-- Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 
-- means foreground (black). 
 
deserialiseHeader :: Get (Word32, Word32, Word32, Word32, [[Word8]])
deserialiseHeader = do
  magicNumber <- getWord32be
  imageCount <- getWord32be
  r <- getWord32be
  c <- getWord32be
  packedData <- getRemainingLazyByteString
  let len = fromIntegral (r * c)
  let unpackedData = S.chunksOf len (BL.unpack packedData)
  return (magicNumber, imageCount, r, c, unpackedData)

readImages :: FilePath -> IO [Image]
readImages filename = do
  content <- GZ.decompress <$> BL.readFile filename
  let (_, _, r, c, unpackedData) = runGet deserialiseHeader content
  return (map (Image (fromIntegral r) (fromIntegral c)) unpackedData)

writeArray :: String -> R.Array R.U R.DIM2 Double -> IO ()
writeArray fileName array = do 
   let (R.Z R.:. r R.:. c) = R.extent array
   B.encodeFile fileName (r,c,R.toList array)

readArray ::String -> IO (R.Array R.U R.DIM2 Double)
readArray fileName = do 
   (r,c,ls) <- B.decodeFile fileName
   return $ R.fromListUnboxed  (R.Z R.:. r R.:. c) ls

generateTrainBatches :: IO ()
generateTrainBatches = do
   images <- readImages "train-images-idx3-ubyte.gz"
   let batches = map toMatrix $ chunksOf 128 images
   (flip mapM_) (zip [0::Integer ..] batches) $ \ (ix, bb) -> do
      let name = "dist/train" ++ (show ix)
      writeArray name bb 

generateTestBatches :: IO ()
generateTestBatches = do
   images <- readImages "t10k-images-idx3-ubyte.gz"
   labels <- readLabels "t10k-labels-idx1-ubyte.gz"
   (flip mapM_) ([0..9]) $ \ ix -> do
      let name = "dist/test" ++ (show ix)
      let batch = filter (((==) ix) . fst) $ zip labels images
      let bb = toMatrix $ snd $ unzip batch 
      writeArray name bb 

generateBigTrainBatches :: IO ()
generateBigTrainBatches = do
   images <- readImages "train-images-idx3-ubyte.gz"
   labels <- readLabels "train-labels-idx1-ubyte.gz"
   (flip mapM_) ([0..9]) $ \ ix -> do
      let name = "dist/bigtrain" ++ (show ix)
      let batch = filter (((==) ix) . fst) $ zip labels images
      let bb = toMatrix $ snd $ unzip batch 
      writeArray name bb 

generateSamples :: IO ()
generateSamples = do
   images <- readImages "train-images-idx3-ubyte.gz"
   labels <- readLabels "train-labels-idx1-ubyte.gz"
   (flip mapM_) ([0..9]) $ \ ix -> do
      gen <- newStdGen
      let name = "dist/sample" ++ (show ix)
      let batch = filter (((==) ix) . fst) $ zip labels images
          batches = snd $ unzip batch
          len = length batches
          rbatches = take 10 $ map (\ rr -> head $ drop rr $ cycle $ batches) (randomRs (0::Int, len - 1) gen)
      let bb = toMatrix $ rbatches 
      writeArray name bb 

printSamples::Int -> String -> RBM.BxI -> IO ()
printSamples imagewidth sfile (RBM.BxI bxi) = do
   let
       computeStrip (Z :. rix :. cix) = 
         let  imagenum = cix `div` imagewidth
              imagepixel = rix * (imagewidth) + (cix `mod` imagewidth)
              pos =  Z :. imagenum :. (imagepixel + 1)
         in   R.rgb8OfGreyDouble $ bxi R.! pos
       rows = R.row $ R.extent bxi
       sh = Z :. imagewidth :. (imagewidth * rows)
   strip <- R.computeUnboxedP $ R.fromFunction sh computeStrip
   R.writeImageToBMP sfile strip

genSample:: String -> DBN.DBN -> IO ()
genSample sname db = do
   let imagewidth = 28
       regenSample ix = do
            let sfile = concat [sname, (show ix), ".bmp"]
            putStrLn $ concat ["generating strip: ", sfile]
            let name = "dist/sample" ++ (show ix)
                readBatch = RBM.BxI <$> (readArray name)
            g1 <- newStdGen
            bxi <- readBatch
            bxh <- DBN.generate g1 db bxi
            g2 <- newStdGen
            bxi' <- DBN.regenerate g2 db bxh
            printSamples imagewidth sfile bxi'
   mapM_ regenSample [0..9::Int] 

mnist :: IO ()
mnist = do 
   gen <- newStdGen
   let [r0,r1,r2] = DBN.dbn gen [785,501,501,11]
       name ix = "dist/train" ++ (show ix)
       readBatch ix = RBM.BxI <$> (readArray (name ix))
       iobatches = map readBatch [0..468::Int]
       p1 = RBM.params { RBM.rate = 0.01, RBM.minMSE = 0.2 }
       p2 = RBM.params { RBM.rate = 0.001, RBM.minMSE = 0.01 }
       
   (head iobatches) >>= (printSamples 28 "dist/original.bmp")
   genSample "dist/strip0." [r0]
   d1 <- DBN.learnLast iobatches p1 [r0]
   genSample "dist/strip1." d1
   d2 <- DBN.learnLast iobatches p2 d1
   genSample "dist/strip2." d2
   d3 <- DBN.learnLast iobatches p1 (d2 ++ [r1])
   genSample "dist/strip3." d3
   d4 <- DBN.learnLast iobatches p2 d3
   genSample "dist/strip4." d4
   d5 <- DBN.learnLast iobatches p1 (d4 ++ [r2])
   genSample "dist/strip5." d5
   d6 <- DBN.learnLast iobatches p2 d5
   genSample "dist/strip6." d6

